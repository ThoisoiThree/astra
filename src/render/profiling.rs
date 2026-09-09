use std::{cell::Cell, sync::mpsc};

pub(super) const PASS_COUNT: usize = 9;
const QUERY_COUNT: u32 = (PASS_COUNT * 2) as u32;
const QUERY_BYTES: u64 = QUERY_COUNT as u64 * std::mem::size_of::<u64>() as u64;

#[derive(Debug, Clone, Copy)]
pub(super) enum ProfilePass {
    Scene = 0,
    AoRaw = 1,
    AoBlur = 2,
    Dof = 3,
    Compose = 4,
    Annotations = 5,
    Ui = 6,
    DofReduce = 7,
    DofSplat = 8,
}

struct PendingReadback {
    buffer: wgpu::Buffer,
    receiver: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    active_mask: u16,
}

pub(super) struct GpuProfiler {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    timestamp_period_ns: f32,
    pending: Option<PendingReadback>,
    latest_ms: [f32; PASS_COUNT],
    latest_frame_ms: f32,
    latest_dof_ms: f32,
    frame_index: u64,
    active_mask: Cell<u16>,
}

impl GpuProfiler {
    pub(super) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| Self {
                query_set: device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("Astra GPU timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: QUERY_COUNT,
                }),
                resolve_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Astra GPU timestamp resolve"),
                    size: QUERY_BYTES,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                timestamp_period_ns: queue.get_timestamp_period(),
                pending: None,
                latest_ms: [0.0; PASS_COUNT],
                latest_frame_ms: 0.0,
                latest_dof_ms: 0.0,
                frame_index: 0,
                active_mask: Cell::new(0),
            })
    }

    pub(super) fn begin_frame(&self) {
        self.active_mask.set(0);
    }

    pub(super) fn writes(&self, pass: ProfilePass) -> wgpu::RenderPassTimestampWrites<'_> {
        self.active_mask
            .set(self.active_mask.get() | (1 << pass as u8));
        let start = pass as u32 * 2;
        wgpu::RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(start),
            end_of_pass_write_index: Some(start + 1),
        }
    }

    pub(super) fn poll(&mut self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::PollType::Poll);
        let ready = self
            .pending
            .as_ref()
            .and_then(|pending| pending.receiver.try_recv().ok());
        let Some(result) = ready else {
            return;
        };
        let Some(pending) = self.pending.take() else {
            return;
        };
        if result.is_ok()
            && let Ok(mapped) = pending.buffer.get_mapped_range(0..QUERY_BYTES)
        {
            let timestamps = bytemuck::cast_slice::<u8, u64>(&mapped);
            // Pass intervals can overlap or include dependency waits. Report
            // elapsed ranges, not a sum that counts the same GPU interval twice.
            self.latest_frame_ms =
                timestamp_span_ms(timestamps, pending.active_mask, self.timestamp_period_ns);
            let dof_mask = (1 << ProfilePass::Dof as u8)
                | (1 << ProfilePass::DofReduce as u8)
                | (1 << ProfilePass::DofSplat as u8);
            self.latest_dof_ms = timestamp_span_ms(
                timestamps,
                pending.active_mask & dof_mask,
                self.timestamp_period_ns,
            );
            for (index, pair) in timestamps.chunks_exact(2).enumerate() {
                self.latest_ms[index] = if pending.active_mask & (1 << index) != 0 {
                    pair[1].saturating_sub(pair[0]) as f32 * self.timestamp_period_ns / 1_000_000.0
                } else {
                    0.0
                };
            }
            drop(mapped);
        }
        pending.buffer.unmap();
    }

    pub(super) fn encode_readback(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Option<wgpu::Buffer> {
        self.frame_index = self.frame_index.wrapping_add(1);
        if self.pending.is_some() || !self.frame_index.is_multiple_of(15) {
            return None;
        }
        encoder.resolve_query_set(&self.query_set, 0..QUERY_COUNT, &self.resolve_buffer, 0);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Astra GPU timestamp readback"),
            size: QUERY_BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&self.resolve_buffer, 0, &readback, 0, QUERY_BYTES);
        Some(readback)
    }

    pub(super) fn begin_readback(&mut self, buffer: wgpu::Buffer) {
        let (sender, receiver) = mpsc::channel();
        buffer.map_async(wgpu::MapMode::Read, 0..QUERY_BYTES, move |result| {
            let _ = sender.send(result);
        });
        self.pending = Some(PendingReadback {
            buffer,
            receiver,
            active_mask: self.active_mask.get(),
        });
    }

    pub(super) fn latest_ms(&self) -> [f32; PASS_COUNT] {
        self.latest_ms
    }

    pub(super) fn latest_frame_ms(&self) -> f32 {
        self.latest_frame_ms
    }
    pub(super) fn latest_dof_ms(&self) -> f32 {
        self.latest_dof_ms
    }
}

fn timestamp_span_ms(timestamps: &[u64], mask: u16, period_ns: f32) -> f32 {
    let mut first = u64::MAX;
    let mut last = 0;
    for (index, pair) in timestamps.chunks_exact(2).enumerate() {
        if mask & (1 << index) != 0 {
            first = first.min(pair[0]);
            last = last.max(pair[1]);
        }
    }
    last.saturating_sub(first) as f32 * period_ns / 1_000_000.0
}
