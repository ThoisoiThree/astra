use std::{ops::Index, sync::Arc};

static TRUE: bool = true;
static FALSE: bool = false;

/// Compact copy-on-write bitset used for per-atom boolean state.
///
/// Cloning is constant-time, which lets an edit transaction remember the old
/// selection without copying one byte per atom. A write detaches only the
/// packed words and keeps unrelated display arrays untouched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomMask {
    len: usize,
    words: Arc<Vec<u64>>,
}

impl Index<usize> for AtomMask {
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
        if self.contains(index) { &TRUE } else { &FALSE }
    }
}

impl AtomMask {
    pub fn new(len: usize, value: bool) -> Self {
        let word_count = len.div_ceil(64);
        let mut words = vec![if value { u64::MAX } else { 0 }; word_count];
        if value && !len.is_multiple_of(64) {
            let valid_bits = len % 64;
            if let Some(last) = words.last_mut() {
                *last = (1_u64 << valid_bits) - 1;
            }
        }
        Self {
            len,
            words: Arc::new(words),
        }
    }

    pub fn from_bools(flags: impl IntoIterator<Item = bool>) -> Self {
        let flags: Vec<bool> = flags.into_iter().collect();
        let mut mask = Self::new(flags.len(), false);
        for (index, value) in flags.into_iter().enumerate() {
            mask.set(index, value);
        }
        mask
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn get(&self, index: usize) -> Option<bool> {
        (index < self.len).then(|| self.contains(index))
    }

    pub fn contains(&self, index: usize) -> bool {
        index < self.len && self.words[index / 64] & (1_u64 << (index % 64)) != 0
    }

    pub fn set(&mut self, index: usize, value: bool) {
        if index >= self.len {
            return;
        }
        let bit = 1_u64 << (index % 64);
        let word = &mut Arc::make_mut(&mut self.words)[index / 64];
        if value {
            *word |= bit;
        } else {
            *word &= !bit;
        }
    }

    pub fn fill(&mut self, value: bool) {
        let len = self.len;
        let words = Arc::make_mut(&mut self.words);
        words.fill(if value { u64::MAX } else { 0 });
        if value && !len.is_multiple_of(64) {
            let valid_bits = len % 64;
            if let Some(last) = words.last_mut() {
                *last = (1_u64 << valid_bits) - 1;
            }
        }
    }

    pub fn count(&self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = bool> + '_ {
        (0..self.len).map(|index| self.contains(index))
    }

    pub fn indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.iter()
            .enumerate()
            .filter_map(|(index, selected)| selected.then_some(index))
    }

    pub fn from_fn(len: usize, mut predicate: impl FnMut(usize) -> bool) -> Self {
        let mut words = vec![0_u64; len.div_ceil(64)];
        for index in 0..len {
            if predicate(index) {
                words[index / 64] |= 1_u64 << (index % 64);
            }
        }
        Self {
            len,
            words: Arc::new(words),
        }
    }

    /// Keeps only bits also set in `other`; lengths must match.
    pub fn intersect_with(&mut self, other: &Self) {
        debug_assert_eq!(self.len, other.len);
        for (word, other) in Arc::make_mut(&mut self.words)
            .iter_mut()
            .zip(other.words.iter())
        {
            *word &= other;
        }
    }

    pub fn union_with(&mut self, other: &Self) {
        debug_assert_eq!(self.len, other.len);
        for (word, other) in Arc::make_mut(&mut self.words)
            .iter_mut()
            .zip(other.words.iter())
        {
            *word |= other;
        }
    }

    pub fn symmetric_difference_with(&mut self, other: &Self) {
        debug_assert_eq!(self.len, other.len);
        for (word, other) in Arc::make_mut(&mut self.words)
            .iter_mut()
            .zip(other.words.iter())
        {
            *word ^= other;
        }
    }

    pub fn invert(&mut self) {
        let len = self.len;
        let words = Arc::make_mut(&mut self.words);
        for word in words.iter_mut() {
            *word = !*word;
        }
        if !len.is_multiple_of(64)
            && let Some(last) = words.last_mut()
        {
            *last &= (1_u64 << (len % 64)) - 1;
        }
    }

    pub fn packed_words(&self) -> &[u64] {
        &self.words
    }

    pub fn estimated_heap_bytes(&self) -> usize {
        self.words.capacity() * size_of::<u64>()
    }
}

#[cfg(test)]
mod tests {
    use super::AtomMask;

    #[test]
    fn masks_pack_bits_and_detach_on_write() {
        let mut mask = AtomMask::new(130, true);
        assert_eq!(mask.count(), 130);
        let unchanged = mask.clone();
        mask.set(64, false);
        assert_eq!(mask.count(), 129);
        assert!(unchanged.contains(64));
        assert!(!mask.contains(64));
        assert_eq!(mask.estimated_heap_bytes(), 24);
    }

    #[test]
    fn set_operations_keep_padding_bits_clear() {
        let evens = AtomMask::from_fn(70, |index| index % 2 == 0);
        let mut inverted = evens.clone();
        inverted.invert();
        assert_eq!(inverted.count(), 35);
        let mut all = evens.clone();
        all.union_with(&inverted);
        assert_eq!(all, AtomMask::new(70, true));
        let mut none = evens.clone();
        none.intersect_with(&inverted);
        assert_eq!(none.count(), 0);
        let mut difference = all.clone();
        difference.symmetric_difference_with(&evens);
        assert_eq!(difference, inverted);
    }
}
