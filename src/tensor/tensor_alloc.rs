use super::*;
use std::alloc::{AllocError, Allocator, Layout};
use std::ptr::NonNull;

unsafe impl Allocator for TensorAllocator {
    #[inline(always)]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let layout = Layout::from_size_align(layout.size() + layout.size() % 16, 16).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            Err(AllocError)
        } else {
            // println!("Pointer address: {:p}", ptr);
            // if ptr as usize % 16 == 0 {
            //     println!("16-byte alignement");
            // }
            let slice = NonNull::slice_from_raw_parts(NonNull::new(ptr).unwrap(), layout.size());
            Ok(slice)
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let layout = Layout::from_size_align(layout.size() + layout.size() % 16, 16).unwrap();
        std::alloc::dealloc(ptr.as_ptr(), layout);
    }
}
