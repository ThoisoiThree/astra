use std::path::PathBuf;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum WindowsBackend {
    #[default]
    Vulkan,
    Dx12,
}

impl WindowsBackend {
    pub fn label(self) -> &'static str {
        match self {
            Self::Vulkan => "Vulkan",
            Self::Dx12 => "DX12",
        }
    }

    pub fn backends(self) -> wgpu::Backends {
        match self {
            Self::Vulkan => wgpu::Backends::VULKAN,
            Self::Dx12 => wgpu::Backends::DX12,
        }
    }

    pub fn settings_path() -> std::io::Result<PathBuf> {
        std::env::var_os("LOCALAPPDATA")
            .map(|base| PathBuf::from(base).join("astra/graphics-backend.txt"))
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::NotFound, "LOCALAPPDATA is unavailable")
            })
    }

    pub fn load() -> Self {
        Self::settings_path()
            .ok()
            .and_then(|path| std::fs::read_to_string(path).ok())
            .filter(|value| value.trim() == "DX12")
            .map_or(Self::Vulkan, |_| Self::Dx12)
    }
}
