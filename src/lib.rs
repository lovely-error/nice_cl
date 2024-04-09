#![feature(generic_arg_infer)]
#![feature(fn_traits)]
#![feature(unboxed_closures)]

mod va_args_emu;


use core::{alloc::Layout, any::TypeId, cell::UnsafeCell, marker::PhantomData, mem::{align_of, align_of_val, forget, size_of, size_of_val, transmute}, ptr::{addr_of, addr_of_mut, copy_nonoverlapping, drop_in_place, null, null_mut}, sync::atomic::{AtomicI32, AtomicU32, AtomicU8, Ordering}};

use cl_sys::{self, c_void, clBuildProgram, clCreateCommandQueue, clCreateContext, clCreateKernel, clCreateProgramWithSource, clEnqueueNDRangeKernel, clGetCommandQueueInfo, clGetDeviceIDs, clGetDeviceInfo, clGetEventInfo, clGetKernelArgInfo, clGetKernelInfo, clGetPlatformInfo, clGetProgramInfo, clReleaseCommandQueue, clReleaseContext, clReleaseDevice, clReleaseEvent, clReleaseKernel, clReleaseProgram, clSVMFree, clSetEventCallback, clSetKernelArg, clSetKernelArgSVMPointer, clWaitForEvents, cl_bitfield, cl_command_queue, cl_command_queue_properties, cl_context, cl_device_id, cl_device_svm_capabilities, cl_event, cl_int, cl_kernel, cl_platform_id, cl_program, cl_uint, libc::c_ulong, size_t, CL_COMPLETE, CL_DEVICE_GLOBAL_MEM_SIZE, CL_DEVICE_MAX_COMPUTE_UNITS, CL_DEVICE_MAX_MEM_ALLOC_SIZE, CL_DEVICE_MAX_WORK_GROUP_SIZE, CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT, CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT, CL_DEVICE_SVM_ATOMICS, CL_DEVICE_SVM_CAPABILITIES, CL_DEVICE_SVM_FINE_GRAIN_BUFFER, CL_DEVICE_SVM_FINE_GRAIN_SYSTEM, CL_DEVICE_TYPE_ALL, CL_DEVICE_VERSION, CL_EVENT_COMMAND_EXECUTION_STATUS, CL_KERNEL_ARG_TYPE_NAME, CL_KERNEL_NUM_ARGS, CL_MEM_READ_WRITE, CL_MEM_SVM_ATOMICS, CL_MEM_SVM_FINE_GRAIN_BUFFER, CL_PLATFORM_VERSION, CL_PROGRAM_KERNEL_NAMES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, CL_QUEUE_PROPERTIES, CL_SUCCESS};

use va_args_emu::{KernelArguments, OpaqueHandle, SomePointer};

#[derive(Debug, Clone, Copy)]
pub enum OCLFailure {
    ResourcesExhausted, NoPlatforms, InvalidProgramm
}
#[derive(Debug, Clone, Copy)]
pub struct MemoryRef<T> {
    ptr: *mut c_void,
    count: usize,
    _phantom: PhantomData<T>
}
impl<T> MemoryRef<T> {
    pub fn len(&self) -> usize { self.count }
    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr.cast()
    }
    pub fn as_single_item(&self) -> &T {
        assert!(self.count == 1);
        unsafe {&*self.ptr.cast()}
    }
    pub fn as_single_mut_item(&mut self) -> &mut T {
        assert!(self.count == 1);
        unsafe {&mut *self.ptr.cast()}
    }
    pub fn as_items(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.ptr.cast(), self.count) }
    }
    pub fn as_mut_items(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr.cast(), self.count) }
    }
}
struct SomeMemoryRef {}
impl<T> va_args_emu::KernelArgument for MemoryRef<T> {
    fn as_opaque(&self) -> OpaqueHandle {
        (addr_of!(self.ptr).cast(), 8, TypeId::of::<SomeMemoryRef>())
    }
}
#[derive(Debug, Clone, Copy)]
pub enum KernelCreationFailure {
    NoMem,
    InvalidArgument(u32),
    InvalidKernelName,
    ArgNumMismatch(u32, u32),
    ArgTypeMismatch(u32)
}
pub struct Kernel {
    handle: cl_kernel
}
impl Drop for Kernel {
    fn drop(&mut self) {
        let _ = unsafe { clReleaseKernel(self.handle) };
    }
}

pub struct CodeBundle {
    handle: cl_program,
    kern_names: Vec<u8>
}
impl CodeBundle {
    pub fn get_available_kernel_names(&self) -> impl Iterator<Item =  &str> {
        let mut last_pivot = 0;
        let names = &self.kern_names;
        let limit = names.len();
        core::iter::from_fn(move || unsafe {
            if last_pivot == limit { return None }
            if names[last_pivot] == ';' as u8 { last_pivot += 1 }
            let mut new_pivot = last_pivot;
            loop {
                new_pivot += 1;
                if new_pivot == limit || names[new_pivot] == ';' as u8 { break }
            }
            let span = &names[last_pivot..new_pivot];
            let str = core::str::from_utf8_unchecked(span);
            last_pivot = new_pivot;
            return Some(str);
        })
    }
    pub fn from_textual_reprs(
        textual_reprs: &[&[u8]],
    ) -> Result<CodeBundle, OCLFailure> { unsafe {
        let input_len = textual_reprs.len();
        let mut str_ptrs = Vec::new();
        str_ptrs.reserve(input_len);
        let mut str_lens = Vec::<size_t>::new();
        str_lens.reserve(input_len);
        for trepr in textual_reprs {
            str_ptrs.push(trepr.as_ptr());
            str_lens.push(trepr.len());
        }
        let ctx = OCL_SHARED_CONTEXT.get_ocl_context();
        let mut ret_code = CL_SUCCESS;
        let cl_prog = clCreateProgramWithSource(
            ctx,
            input_len as _,
            str_ptrs.as_ptr().cast(),
            str_lens.as_ptr(),
            &mut ret_code
        );
        match ret_code {
            cl_sys::CL_SUCCESS => (),
            cl_sys::CL_OUT_OF_RESOURCES |
            cl_sys::CL_OUT_OF_HOST_MEMORY => {
                return Err(OCLFailure::ResourcesExhausted)
            },
            cl_sys::CL_INVALID_VALUE |
            cl_sys::CL_INVALID_CONTEXT |
            _ => unreachable!()
        }
        let comp_args = "-cl-no-signed-zeros -cl-std=CL2.0 -cl-kernel-arg-info -O2\0";
        let devs = OCL_SHARED_CONTEXT.dev_ids.as_ptr();
        let devs_len = OCL_SHARED_CONTEXT.len;
        let ret_code = clBuildProgram(
            cl_prog,
            devs_len,
            devs,
            comp_args.as_ptr().cast(),
            None,
            null_mut()
        );
        match ret_code {
            cl_sys::CL_SUCCESS => (),
            cl_sys::CL_OUT_OF_RESOURCES |
            cl_sys::CL_OUT_OF_HOST_MEMORY => {
                return Err(OCLFailure::ResourcesExhausted)
            },
            cl_sys::CL_BUILD_PROGRAM_FAILURE |
            cl_sys::CL_INVALID_PROGRAM => {
                return Err(OCLFailure::InvalidProgramm)
            },
            cl_sys::CL_INVALID_BUILD_OPTIONS => {
                panic!("Build options invalid! {}", comp_args)
            },
            cl_sys::CL_COMPILER_NOT_AVAILABLE => {
                panic!("No OCL compiler available!")
            },
            cl_sys::CL_INVALID_OPERATION |
            cl_sys::CL_INVALID_BINARY |
            _ => unreachable!()
        }
        let mut kern_name_bytes = Vec::<u8>::new();
        kern_name_bytes.reserve(64);
        let mut len = 0;
        loop {
            let ret_code = clGetProgramInfo(
                cl_prog,
                CL_PROGRAM_KERNEL_NAMES,
                kern_name_bytes.capacity(),
                kern_name_bytes.as_mut_ptr().cast(),
                &mut len
            );
            match ret_code {
                cl_sys::CL_SUCCESS => break,
                cl_sys::CL_INVALID_VALUE => {
                    kern_name_bytes.reserve(64);
                    continue;
                },
                cl_sys::CL_OUT_OF_RESOURCES |
                cl_sys::CL_OUT_OF_HOST_MEMORY => {
                    return Err(OCLFailure::ResourcesExhausted)
                },
                cl_sys::CL_INVALID_PROGRAM_EXECUTABLE |
                cl_sys::CL_INVALID_PROGRAM |
                _ => unreachable!()
            }
        }
        kern_name_bytes.set_len(len);

        let val = CodeBundle {
            handle: cl_prog,
            kern_names: kern_name_bytes
        };
        return Ok(val);
    } }
    pub fn build_kernel(
        &self,
        name: &str,
        args: impl KernelArguments
    ) -> Result<Kernel, KernelCreationFailure> { unsafe {
        let name = format!("{}\0", name);
        let mut ret_code = CL_SUCCESS;
        let kern_ptr = clCreateKernel(
            self.handle,
            name.as_ptr().cast(),
            &mut ret_code
        );
        match ret_code {
            cl_sys::CL_SUCCESS => (),
            cl_sys::CL_OUT_OF_RESOURCES |
            cl_sys::CL_OUT_OF_HOST_MEMORY => {
                return Err(KernelCreationFailure::NoMem)
            },
            cl_sys::CL_INVALID_KERNEL_NAME => {
                return Err(KernelCreationFailure::InvalidKernelName)
            }
            cl_sys::CL_INVALID_VALUE |
            cl_sys::CL_INVALID_PROGRAM |
            cl_sys::CL_INVALID_PROGRAM_EXECUTABLE |
            _ => unreachable!()
        }
        let mut arg_count = 0u32;
        let ret_code = clGetKernelInfo(
            kern_ptr,
            CL_KERNEL_NUM_ARGS,
            size_of::<cl_uint>(),
            addr_of_mut!(arg_count).cast(),
            null_mut()
        );
        match ret_code {
            cl_sys::CL_SUCCESS => (),
            cl_sys::CL_OUT_OF_RESOURCES |
            cl_sys::CL_OUT_OF_HOST_MEMORY => {
                return Err(KernelCreationFailure::NoMem)
            },
            cl_sys::CL_INVALID_VALUE |
            cl_sys::CL_INVALID_KERNEL |
            _ => unreachable!()
        }
        let expected = arg_count;
        let actual = args.len() as u32;
        if actual != expected {
            return Err(KernelCreationFailure::ArgNumMismatch(expected, actual));
        }
        let mut iter = args.iter();
        let mut ix = 0;
        let mut arg_ty_nm = [0u8;16];
        let mut arg_ty_nm_len = 0;
        while let Some((ptr, size, id)) = iter.next() {
            let ret_code = clGetKernelArgInfo(
                kern_ptr,
                ix,
                CL_KERNEL_ARG_TYPE_NAME,
                16,
                arg_ty_nm.as_mut_ptr().cast(),
                &mut arg_ty_nm_len
            );
            match ret_code {
                cl_sys::CL_SUCCESS => (),
                cl_sys::CL_KERNEL_ARG_INFO_NOT_AVAILABLE => {
                    panic!("Kernel function type information is not present in the binary!")
                },
                _ => unreachable!()
            }
            let slice = core::slice::from_raw_parts(arg_ty_nm.as_ptr(), arg_ty_nm_len);
            let str = core::str::from_utf8_unchecked(slice);
            let is_pointer = str.contains('*');
            if is_pointer {
                let okay = id == TypeId::of::<SomePointer>() || id == TypeId::of::<SomeMemoryRef>();
                if !okay {
                    return Err(KernelCreationFailure::ArgTypeMismatch(ix));
                }
            } else {
                match str {
                    "char\0" => {
                        if id != TypeId::of::<i8>() {
                            return Err(KernelCreationFailure::ArgTypeMismatch(ix));
                        }
                    },
                    "uchar\n" | "unsigned char\0" => {
                        if id != TypeId::of::<u8>() {
                            return Err(KernelCreationFailure::ArgTypeMismatch(ix));
                        }
                    }
                    "short\0" => {
                        if id != TypeId::of::<i16>() {
                            return Err(KernelCreationFailure::ArgTypeMismatch(ix));
                        }
                    },
                    "ushort\0" | "unsigned short\0" => {
                        if id != TypeId::of::<u16>() {
                            return Err(KernelCreationFailure::ArgTypeMismatch(ix));
                        }
                    },
                    "int\0" => {
                        if id != TypeId::of::<i32>() {
                            return Err(KernelCreationFailure::ArgTypeMismatch(ix));
                        }
                    },
                    "uint\0" | "unsigned int\0" => {
                        if id != TypeId::of::<u32>() {
                            return Err(KernelCreationFailure::ArgTypeMismatch(ix));
                        }
                    },
                    "long\0" => {
                        if id != TypeId::of::<i64>() {
                            return Err(KernelCreationFailure::ArgTypeMismatch(ix));
                        }
                    },
                    "ulong\0" | "unsigned long\0" => {
                        if id != TypeId::of::<u64>() {
                            return Err(KernelCreationFailure::ArgTypeMismatch(ix));
                        }
                    },
                    _ => unreachable!()
                }
            }
            let ret_c ;
            match id {
                _ if id == TypeId::of::<SomeMemoryRef>() => {
                    let ptr = ptr.cast::<*const c_void>().read();
                    ret_c = clSetKernelArgSVMPointer(kern_ptr, ix, ptr);
                },
                _ => {
                    ret_c = clSetKernelArg(kern_ptr, ix, size, ptr.cast());
                }
            }
            match ret_c {
                cl_sys::CL_SUCCESS => (),
                cl_sys::CL_OUT_OF_RESOURCES |
                cl_sys::CL_OUT_OF_HOST_MEMORY => {
                    return Err(KernelCreationFailure::NoMem)
                },
                cl_sys::CL_INVALID_DEVICE_QUEUE => {
                    panic!("Invalid device queue!")
                },
                cl_sys::CL_INVALID_ARG_INDEX |
                cl_sys::CL_INVALID_ARG_VALUE |
                cl_sys::CL_INVALID_ARG_SIZE |
                cl_sys::CL_INVALID_MEM_OBJECT |
                cl_sys::CL_INVALID_SAMPLER => {
                    return Err(KernelCreationFailure::InvalidArgument(ix));
                }
                cl_sys::CL_INVALID_KERNEL |
                _ => unreachable!()
            }
            ix += 1;
        }

        let res = Kernel {
            handle: kern_ptr
        };
        return Ok(res)
    } }
}
impl Drop for CodeBundle {
    fn drop(&mut self) {
        let _ = unsafe { clReleaseProgram(self.handle) };
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CompletionAwaitFailure {
    NoMem, JobFinishedWithError
}
#[derive(Debug, Clone, Copy)]
pub enum ExecutionState {
    Queued, Submited, Running, Complete
}


struct TokenInner {
    token: cl_event,
    file_desc: i32,
    futex: i32
}
pub struct Token(UnsafeCell<TokenInner>);
impl Token {
    pub fn await_completion(&self) -> Result<(), CompletionAwaitFailure> { unsafe {
        let this = &mut *self.0.get();
        match clWaitForEvents(1, &this.token) {
            cl_sys::CL_SUCCESS => (),
            cl_sys::CL_OUT_OF_RESOURCES |
            cl_sys::CL_OUT_OF_HOST_MEMORY => {
                return Err(CompletionAwaitFailure::NoMem)
            },
            cl_sys::CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST => {
                return Err(CompletionAwaitFailure::JobFinishedWithError);
            }
            cl_sys::CL_INVALID_EVENT |
            cl_sys::CL_INVALID_VALUE |
            cl_sys::CL_INVALID_CONTEXT |
            _ => unreachable!()
        }
        return Ok(());
    } }
    pub fn get_execution_state(&self) -> Result<ExecutionState, OCLFailure> { unsafe {
        let this = &mut *self.0.get();
        let mut value = 0;
        let ret_code = clGetEventInfo(
            this.token,
            CL_EVENT_COMMAND_EXECUTION_STATUS,
            size_of::<cl_int>(),
            addr_of_mut!(value).cast(),
            null_mut()
        );
        match ret_code {
            cl_sys::CL_SUCCESS => (),
            cl_sys::CL_OUT_OF_RESOURCES |
            cl_sys::CL_OUT_OF_HOST_MEMORY => {
                return Err(OCLFailure::ResourcesExhausted)
            },
            _ => unreachable!()
        }
        let val = match value {
            cl_sys::CL_QUEUED => ExecutionState::Queued,
            cl_sys::CL_SUBMITTED => ExecutionState::Submited,
            cl_sys::CL_RUNNING => ExecutionState::Running,
            cl_sys::CL_COMPLETE => ExecutionState::Complete,
            _ => unreachable!()
        };

        return Ok(val);
    } }
    pub fn attach_completion_callback<F: FnMut(cl_int) -> ()>(
        &self,
        action: F
    ) -> Result<(), OCLFailure> { unsafe {
        #[repr(C)]
        struct Metadata {
            invocation_fun: fn (*mut (), i32),
            capture_destructor_fun: unsafe fn (*mut ()),
            size: u32,
            back_offset: u32
        }
        let capture_size = size_of_val(&action);
        let mut closure_size = size_of::<Metadata>();
        closure_size = closure_size.next_multiple_of(align_of_val(&action));
        let pivot = closure_size;
        closure_size += capture_size;

        let mem_origin = std::alloc::alloc(Layout::from_size_align_unchecked(closure_size, 8));
        let pivoted_mem = mem_origin.byte_add(pivot);

        let inv_fun = <F as FnMut<(cl_int,)>>::call_mut;
        let inv_fun = transmute::<_, fn (*mut (), i32)>(inv_fun as *mut ());
        let dctor = drop_in_place::<F>;
        let dctor = transmute::<_, unsafe fn(*mut ())>(dctor as *mut ());
        let mtd = Metadata {
            invocation_fun: inv_fun,
            capture_destructor_fun: dctor,
            size: closure_size as u32,
            back_offset: pivot as u32
        };
        let mtd_ptr = pivoted_mem.cast::<Metadata>().sub(1);
        mtd_ptr.write(mtd);

        let capture_src_ptr = addr_of!(action).cast::<u8>();
        let capture_dst_ptr = pivoted_mem;
        copy_nonoverlapping(capture_src_ptr, capture_dst_ptr, capture_size);
        forget(action);

        let user_data_ptr = pivoted_mem.cast::<c_void>();

        unsafe extern "C" fn invoker(_:cl_event, code:cl_int, ud: *mut ()) {
            let Metadata { invocation_fun, capture_destructor_fun, size, back_offset } =
                *ud.cast::<Metadata>().sub(1);
            let capture_ptr = ud;
            invocation_fun(capture_ptr, code);
            capture_destructor_fun(capture_ptr);
            let origin_ptr = ud.cast::<u8>().byte_sub(back_offset as _);
            std::alloc::dealloc(origin_ptr, Layout::from_size_align_unchecked(size as _, 8));
        }
        type ClCallback = extern "C" fn(cl_event, cl_int, *mut c_void) ;
        let fptr = transmute::<_, ClCallback>(invoker as *mut ());

        let this = &mut *self.0.get();
        let ret_code = clSetEventCallback(
            this.token,
            CL_COMPLETE,
            Some(fptr),
            user_data_ptr
        );
        match ret_code {
            cl_sys::CL_SUCCESS => (),
            cl_sys::CL_OUT_OF_RESOURCES |
            cl_sys::CL_OUT_OF_HOST_MEMORY => {
                return Err(OCLFailure::ResourcesExhausted)
            },
            _ => unreachable!()
        }
        return Ok(());
    } }
    // pub fn as_fd(&self) -> Result<i32, OCLFailure> { unsafe {
    //     let this = &mut *self.0.get();
    //     if this.file_desc != -1 {
    //         return Ok(this.file_desc);
    //     }
    //     let fd_nm = "__OCL_WAIT_TOKEN__\0";
    //     let fd = libc::memfd_create(fd_nm.as_ptr().cast(), 0);
    //     if fd == -1 {
    //         match *libc::__errno_location() {
    //             libc::ENOMEM |
    //             libc::ENFILE |
    //             libc::EMFILE => {
    //                 return Err(OCLFailure::ResourcesExhausted);
    //             },
    //             libc::EFAULT |
    //             libc::EINVAL |
    //             libc::EPERM |
    //             _ => unreachable!()
    //         }
    //     }
    //     this.file_desc = fd;
    //     unsafe extern "C" fn invoker(_:cl_event, _:cl_int, ud: *mut ()) {
    //         let fd = ud as i32;
    //         let str = "done";
    //         loop {
    //             let outcome = libc::write(fd, str.as_ptr().cast(), str.len());
    //             if outcome == -1 {
    //                 match *libc::__errno_location() {
    //                     libc:: ENOSPC |
    //                     libc::EDQUOT => {
    //                         panic!("Resources exhausted")
    //                     }
    //                     libc::EPIPE |
    //                     libc::EPERM |
    //                     libc::EIO |
    //                     libc::EINVAL |
    //                     libc::EINTR => continue,
    //                     libc::EFBIG |
    //                     libc::EFAULT |
    //                     libc::EDESTADDRREQ |
    //                     libc::EBADF |
    //                     libc::EWOULDBLOCK |
    //                     _ => unreachable!()
    //                 }
    //             }
    //             break;
    //         }
    //     }
    //     type ClCallback = extern "C" fn(cl_event, cl_int, *mut c_void) ;
    //     let fptr = transmute::<_, ClCallback>(invoker as *mut ());
    //     let ret_code = clSetEventCallback(
    //         this.token,
    //         CL_COMPLETE,
    //         Some(fptr),
    //         fd as _
    //     );
    //     match ret_code {
    //         cl_sys::CL_SUCCESS => (),
    //         cl_sys::CL_OUT_OF_RESOURCES |
    //         cl_sys::CL_OUT_OF_HOST_MEMORY => {
    //             return Err(OCLFailure::ResourcesExhausted)
    //         },
    //         _ => unreachable!()
    //     }

    //     return Ok(fd);
    // } }
    pub fn as_futex(&self) -> Result<&AtomicI32, OCLFailure> { unsafe {
        let this = &mut *self.0.get();
        let ptr = addr_of_mut!(this.futex).cast::<AtomicI32>();
        let fref = &*ptr;
        match fref.compare_exchange(1, 2, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => (),
            Err(_) => return Ok(fref),
        }
        unsafe extern "C" fn invoker(_:cl_event, ret_code:cl_int, ud: *mut AtomicI32) {
            let futex = &*ud;
            futex.store(ret_code, Ordering::Relaxed);
            libc::syscall(
                libc::SYS_futex,
                futex,
                libc::FUTEX_WAKE,
                u32::MAX,
                0,
                0
            );
        }
        type ClCallback = extern "C" fn(cl_event, cl_int, *mut c_void) ;
        let fptr = transmute::<_, ClCallback>(invoker as *mut ());
        let ret_code = clSetEventCallback(
            this.token,
            CL_COMPLETE,
            Some(fptr),
            ptr.cast()
        );
        match ret_code {
            cl_sys::CL_SUCCESS => (),
            cl_sys::CL_OUT_OF_RESOURCES |
            cl_sys::CL_OUT_OF_HOST_MEMORY => {
                return Err(OCLFailure::ResourcesExhausted)
            },
            _ => unreachable!()
        }
        return Ok(fref);
    } }
    pub fn await_completion_on_token_futex(
        token_futex: &AtomicI32
    ) { unsafe {
        let _ = libc::syscall(
            libc::SYS_futex,
            token_futex,
            libc::FUTEX_WAIT,
            2,
            0,
            0
        );
    } }
}
impl Drop for Token {
    fn drop(&mut self) { unsafe {
        let this = &mut *self.0.get();
        let _ = clReleaseEvent(this.token);
        if this.file_desc != -1 {
            loop {
                let outcome = libc::close(this.file_desc);
                if outcome == -1 {
                    match *libc::__errno_location() {
                        libc:: EINTR => continue,
                        libc::EDQUOT |
                        libc::ENOSPC |
                        libc::EIO |
                        libc::EBADF |
                        _ => unreachable!()
                    }
                }
                break;
            }
        }
    } }
}

static mut DROP_COUNT: u8 = 0;
#[derive(Debug)]
struct Test;
impl Drop for Test {
    fn drop(&mut self) {
        unsafe {DROP_COUNT += 1}
    }
}
#[test] #[ignore]
fn clos() {
    fn test<F: FnMut()>(
        mut action: F
    ) {
        println!("size {}", size_of_val(&action));
        println!("align {}", align_of_val(&action));

        // let ptr = addr_of!(action).cast::<[u64;8]>();
        // println!("{:#?}", unsafe{*ptr});

        let proc = <F as FnMut<()>>::call_mut;
        let () = proc(&mut action, ());

        let trivial = drop_in_place::<F>;
        // println!("{:?}", trivial as *mut ())
        unsafe { trivial(&mut action) };
        forget(action);
    }
    let d = (133u64,Test);
    test(move || println!("{:?}", d));
    assert!(unsafe {DROP_COUNT == 1});
}

#[derive(Debug, Clone, Copy)]
pub enum KernelLaunchFailure {
    NoMem, InvalidArgs
}
pub struct Device {
    ext: Box<DeviceSpecificExtData>
}
impl Device {
    pub fn allocate_memory<T>(&self, count: usize) -> Result<MemoryRef<T>, OCLFailure> { unsafe {
        assert!(count > 0, "Item count cannot be zero");
        assert!(self.ext.props.shared_mem_caps.fine_grain_buffer, "No support of fine-grain buffers!");
        let alloc_props =
            CL_MEM_READ_WRITE |
            CL_MEM_SVM_FINE_GRAIN_BUFFER |
            if self.ext.props.shared_mem_caps.svm_atomics { CL_MEM_SVM_ATOMICS } else { 0 };
        let align =
            align_of::<T>()
            .max(self.ext.props.shared_mem_caps.preffered_platform_atomic_alignment as _);
        let size = size_of::<T>() * count;
        let ctx = OCL_SHARED_CONTEXT.get_ocl_context();
        let ptr = cl_sys::clSVMAlloc(
            ctx,
            alloc_props,
            size,
            align as _
        );
        if ptr == null_mut() {
            return Err(OCLFailure::ResourcesExhausted);
        }
        let ret = MemoryRef {
            ptr: ptr,
            count: count,
            _phantom: PhantomData
        };
        return Ok(ret);
    } }
    pub fn deallocate_memory<T>(&self, memory_ref: MemoryRef<T>) { unsafe {
        let ctx = OCL_SHARED_CONTEXT.get_ocl_context();
        let () = clSVMFree(ctx, memory_ref.ptr);
    } }
    pub fn launch_kernel(
        &self,
        kernel: Kernel,
        grid_dims: usize
    ) -> Result<Token, KernelLaunchFailure> { unsafe {
        let grid_dim = 1;
        let dims: [size_t;3] = [grid_dims, 0, 0];
        let mut completion_token = null_mut();
        let ret_code = clEnqueueNDRangeKernel(
            self.ext.command_queue,
            kernel.handle,
            grid_dim,
            null(),
            dims.as_ptr(),
            null(),
            0,
            null(),
            &mut completion_token
        );
        match ret_code {
            cl_sys::CL_SUCCESS => (),
            cl_sys::CL_OUT_OF_RESOURCES |
            cl_sys::CL_MEM_OBJECT_ALLOCATION_FAILURE |
            cl_sys::CL_OUT_OF_HOST_MEMORY => {
                return Err(KernelLaunchFailure::NoMem)
            },
            cl_sys::CL_INVALID_WORK_ITEM_SIZE |
            cl_sys::CL_INVALID_WORK_GROUP_SIZE |
            cl_sys::CL_INVALID_GLOBAL_OFFSET |
            cl_sys::CL_INVALID_GLOBAL_WORK_SIZE |
            cl_sys::CL_INVALID_KERNEL_ARGS |
            cl_sys::CL_INVALID_EVENT_WAIT_LIST |
            cl_sys::CL_INVALID_OPERATION |
            cl_sys::CL_IMAGE_FORMAT_NOT_SUPPORTED |
            cl_sys::CL_INVALID_IMAGE_SIZE |
            cl_sys::CL_MISALIGNED_SUB_BUFFER_OFFSET |
            cl_sys::CL_INVALID_WORK_DIMENSION => {
                return Err(KernelLaunchFailure::InvalidArgs)
            },
            _ => unreachable!()
        }
        let tok = Token(UnsafeCell::new(TokenInner {
            token: completion_token,
            file_desc: -1,
            futex: 1
        }));
        return Ok(tok);
    } }
    pub fn get_properties(&self) -> DeviceProps {
        self.ext.props
    }
}
impl Drop for Device {
    fn drop(&mut self) { unsafe {
        let _ = clReleaseCommandQueue(self.ext.command_queue);
        let _ = clReleaseDevice(self.ext.handle);
        let prior = OCL_SHARED_CONTEXT.dev_refs.fetch_sub(1, Ordering::Release);
        if prior == 1 {
            let _ = clReleaseContext(OCL_SHARED_CONTEXT.cl_contex);
            OCL_SHARED_CONTEXT.state.store(OCLSharedContextLifecycleState::Uninit as _, Ordering::Release);
        }
    } }
}
#[derive(Debug, Clone, Copy)]
pub struct DeviceProps {
    pub compute_unit_count: u32,
    pub max_work_group_size: usize,
    pub max_alloc_size_in_bytes: usize,
    pub global_mem_size: usize,
    pub shared_mem_caps: DeviceSVMProps,
    pub main_queue_is_async: bool,
    pub supported_cl_version: (u8,u8)
}
struct DeviceSpecificExtData {
    command_queue: cl_command_queue,
    handle: cl_device_id,
    props: DeviceProps
}
#[derive(Debug, Clone, Copy)]
pub struct DeviceSVMProps {
    pub fine_grain_buffer: bool,
    pub fine_grain_system: bool,
    pub svm_atomics: bool,
    pub preffered_platform_atomic_alignment: u32,
    pub preffered_platform_global_alignment: u32
}

pub struct Platform {
    handle: cl_platform_id,
    name: String,
    version: (u8,u8)
}
impl Platform {
    pub fn get_ocl_version(&self) -> (u8, u8) {
        self.version
    }
    pub fn get_platform_name(&self) -> &String {
        return &self.name;
    }
}

fn try_get_platform_name(
    scratch: *mut u8,
    size: usize,
    handle: cl_platform_id
) -> Option<Result<String, OCLFailure>> { unsafe {

    let mut len = 0;
    let ret_code = clGetPlatformInfo(
        handle,
        CL_PLATFORM_VERSION,
        size,
        scratch.cast(),
        &mut len
    );
    match ret_code {
        cl_sys::CL_SUCCESS => (),
        cl_sys::CL_OUT_OF_HOST_MEMORY => {
            return Some(Err(OCLFailure::ResourcesExhausted))
        },
        cl_sys::CL_INVALID_VALUE => return None,
        _ => unreachable!()
    }
    let slice = core::slice::from_raw_parts(scratch, len);
    let str = core::str::from_utf8_unchecked(slice).to_string();

    return Some(Ok(str));
} }

fn try_get_version(
    scratch: *mut u8,
    size: usize,
    handle: cl_platform_id
) -> Option<Result<(u8,u8), OCLFailure>> { unsafe {
    let mut len = 0;
    let ret_code = clGetPlatformInfo(
        handle,
        CL_PLATFORM_VERSION,
        size,
        scratch.cast(),
        &mut len
    );
    match ret_code {
        cl_sys::CL_SUCCESS => (),
        cl_sys::CL_OUT_OF_HOST_MEMORY => {
            return Some(Err(OCLFailure::ResourcesExhausted))
        }
        cl_sys::CL_INVALID_VALUE => return None, // we nee more mem, ugh
        _ => unreachable!()
    }
    let slice = core::slice::from_raw_parts(scratch, len);
    let maj = slice[7] - 48;
    let min = slice[9] - 48;

    return Some(Ok((maj, min)));
} }
pub fn get_current_platform<'a>() -> &'a Platform { unsafe {
    while !OCL_SHARED_CONTEXT.is_ready() {}
    match OCL_SHARED_CONTEXT.platform.as_ref() {
        Some(ptr) => ptr,
        None => panic!("No platform was set. Call set_default_platform or enumerate_devices"),
    }
} }
pub fn set_default_platform(
    platform: Platform
) {
    unsafe { OCL_SHARED_CONTEXT.platform.replace(platform) };
}
pub fn enumerate_platforms() -> Result<Vec<Platform>, OCLFailure> { unsafe {

    const NUM: u32 = 8;
    let mut platforms: [cl_platform_id;NUM as _] = [null_mut();NUM as _];
    let mut num_present_platforms = 0;

    let ret_code = cl_sys::clGetPlatformIDs(
        NUM,
        platforms.as_mut_ptr().cast::<cl_platform_id>(),
        &mut num_present_platforms
    );
    match ret_code {
        cl_sys::CL_SUCCESS => (),
        cl_sys::CL_PLATFORM_NOT_FOUND_KHR => {
            return Err(OCLFailure::NoPlatforms);
        },
        cl_sys::CL_OUT_OF_HOST_MEMORY => {
            return Err(OCLFailure::ResourcesExhausted)
        },
        cl_sys::CL_INVALID_VALUE |
        _ => unreachable!()
    }
    if num_present_platforms == 0 {
        return Err(OCLFailure::NoPlatforms);
    }

    let mut pfs = Vec::new();
    pfs.reserve(num_present_platforms as _);

    let mut size = 4096;
    let mut meta = Layout::from_size_align_unchecked(size, 1);
    let mut scratch = std::alloc::alloc(meta);

    for i in 0 .. num_present_platforms {
        let pf = platforms[i as usize];
        let ver = loop {
            match try_get_version(scratch, size, pf) {
                Some(thing) => break thing?,
                None => {
                    size *= 2;
                    scratch = std::alloc::realloc(scratch, meta, size);
                    meta = Layout::from_size_align_unchecked(size, 1);
                },
            }
        };
        let nm = loop {
            match try_get_platform_name(scratch, size, pf) {
                Some(thing) => break thing?,
                None => {
                    size *= 2;
                    scratch = std::alloc::realloc(scratch, meta, size);
                    meta = Layout::from_size_align_unchecked(size, 1);
                },
            }
        };
        let pf = Platform {
            handle: pf,
            version: ver,
            name: nm,
        };
        pfs.push(pf)
    }
    std::alloc::dealloc(scratch, meta);



    return Ok(pfs)
} }
#[derive(Debug, Clone, Copy)] #[repr(u8)]
enum OCLSharedContextLifecycleState {
    Uninit, InInit, DoneInit
}
struct OCLSharedContext {
    cl_contex: cl_context,
    dev_ids: [cl_device_id; 16],
    dev_refs: AtomicU32,
    len: u32,
    state: AtomicU8,
    platform: Option<Platform>,
}
impl OCLSharedContext {
    fn is_ready(&self) -> bool {
        let outcome = self.state.compare_exchange(
            OCLSharedContextLifecycleState::DoneInit as _,
            OCLSharedContextLifecycleState::DoneInit as _,
            Ordering::Acquire,
            Ordering::Relaxed
        );
        match outcome {
            Ok(_) => true,
            Err(real) => match unsafe { transmute::<_, OCLSharedContextLifecycleState>(real) } {
                OCLSharedContextLifecycleState::Uninit => panic!("Attempt to access uninited ocl context!"),
                OCLSharedContextLifecycleState::InInit => false,
                OCLSharedContextLifecycleState::DoneInit => unreachable!(),
            },
        }
    }
    fn get_ocl_context(&self) -> cl_context {
        while !self.is_ready() {}
        self.cl_contex
    }
}
static mut OCL_SHARED_CONTEXT: OCLSharedContext = OCLSharedContext {
    cl_contex: null_mut(),
    dev_ids: [null_mut();_],
    len: 0,
    dev_refs: AtomicU32::new(0),
    state: AtomicU8::new(OCLSharedContextLifecycleState::Uninit as _),
    platform: None,
};

pub fn enumerate_devices() -> Result<Vec<Device>, OCLFailure> { unsafe {

    let outcome = OCL_SHARED_CONTEXT.state.compare_exchange(
        OCLSharedContextLifecycleState::Uninit as _,
        OCLSharedContextLifecycleState::InInit as _,
        Ordering::Relaxed,
        Ordering::Relaxed
    );
    match outcome {
        Ok(_) => (),
        Err(real) => match transmute::<_, OCLSharedContextLifecycleState>(real) {
            OCLSharedContextLifecycleState::Uninit => unreachable!(),
            OCLSharedContextLifecycleState::InInit |
            OCLSharedContextLifecycleState::DoneInit => panic!("Attempt to reinit OCL context"),
        },
    }

    if OCL_SHARED_CONTEXT.platform.is_none() {
        let pfs = enumerate_platforms()?;
        let pfs = pfs.into_iter().filter(|i| i.get_ocl_version().0 >= 2);
        let mut pfs = pfs.collect::<Vec<_>>();
        if pfs.len() > 1 {
            panic!("Cant pick one platform. Several are present! Use set_default_platform to pick yourself")
        }
        let pf = pfs.pop().unwrap();
        OCL_SHARED_CONTEXT.platform = Some(pf);
    }

    let pfh = OCL_SHARED_CONTEXT.platform.as_ref().unwrap().handle;

    let mut dev_ids : [cl_device_id;16] = [null_mut(); _];
    let mut len = 0;
    let ret_code = clGetDeviceIDs(
        pfh,
        CL_DEVICE_TYPE_ALL,
        16,
        dev_ids.as_mut_ptr(),
        &mut len
    );
    match ret_code {
        cl_sys::CL_SUCCESS => (),
        cl_sys::CL_DEVICE_NOT_FOUND => (),
        cl_sys::CL_OUT_OF_RESOURCES |
        cl_sys::CL_OUT_OF_HOST_MEMORY => {
            return Err(OCLFailure::ResourcesExhausted)
        },
        cl_sys::CL_INVALID_PLATFORM |
        cl_sys::CL_INVALID_DEVICE_TYPE |
        cl_sys::CL_INVALID_VALUE |
        _ => unreachable!(),
    }
    len = len.min(16);

    OCL_SHARED_CONTEXT.dev_ids = dev_ids;
    OCL_SHARED_CONTEXT.len = len;
    OCL_SHARED_CONTEXT.dev_refs.store(len, Ordering::Relaxed);

    let mut devs = Vec::new();
    devs.reserve(len as _);

    for i in 0 .. len {
        let dev_han = dev_ids[i as usize];

        let mut ret_c  = CL_SUCCESS;
        let mut cu_num = 0;
        ret_c |= clGetDeviceInfo(
            dev_han,
            CL_DEVICE_MAX_COMPUTE_UNITS,
            size_of::<cl_uint>(),
            addr_of_mut!(cu_num).cast(),
            null_mut()
        );
        let mut wg_max_size = 0;
        ret_c |= clGetDeviceInfo(
            dev_han,
            CL_DEVICE_MAX_WORK_GROUP_SIZE,
            size_of::<size_t>(),
            addr_of_mut!(wg_max_size).cast(),
            null_mut()
        );
        let mut max_alloc_size = 0;
        ret_c |= clGetDeviceInfo(
            dev_han,
            CL_DEVICE_MAX_MEM_ALLOC_SIZE,
            size_of::<c_ulong>(),
            addr_of_mut!(max_alloc_size).cast(),
            null_mut()
        );
        let mut global_mem_size = 0;
        ret_c |= clGetDeviceInfo(
            dev_han,
            CL_DEVICE_GLOBAL_MEM_SIZE,
            size_of::<c_ulong>(),
            addr_of_mut!(global_mem_size).cast(),
            null_mut()
        );
        let mut svm_caps = 0;
        ret_c |= clGetDeviceInfo(
            dev_han,
            CL_DEVICE_SVM_CAPABILITIES,
            size_of::<cl_device_svm_capabilities>(),
            addr_of_mut!(svm_caps).cast(),
            null_mut()
        );
        let mut svm_atomic_platform_align = 0u32;
        ret_c |= clGetDeviceInfo(
            dev_han,
            CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT,
            size_of::<cl_uint>(),
            addr_of_mut!(svm_atomic_platform_align).cast(),
            null_mut()
        );
        let mut svm_atomic_global_align = 0u32;
        ret_c |= clGetDeviceInfo(
            dev_han,
            CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT,
            size_of::<cl_uint>(),
            addr_of_mut!(svm_atomic_global_align).cast(),
            null_mut()
        );
        let mut version_str = [0u8;128];
        let mut actual = 0;
        ret_c |= clGetDeviceInfo(
            dev_han,
            CL_DEVICE_VERSION,
            128,
            addr_of_mut!(version_str).cast(),
            &mut actual
        );
        let cl_version = (version_str[7] - 48, version_str[9] - 48);
        if ret_c != cl_sys::CL_SUCCESS {
            let no_mem =
                ret_c & cl_sys::CL_OUT_OF_RESOURCES != 0 ||
                ret_c & cl_sys::CL_OUT_OF_HOST_MEMORY != 0;
            if no_mem {
                return Err(OCLFailure::ResourcesExhausted)
            }
            let invalid =
                ret_c & cl_sys::CL_INVALID_DEVICE != 0 ||
                ret_c & cl_sys::CL_INVALID_VALUE != 0;
            if invalid {
                panic!()
            }
        }
        let svm_caps = DeviceSVMProps {
            fine_grain_buffer: svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER != 0,
            fine_grain_system: svm_caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM != 0,
            svm_atomics: svm_caps & CL_DEVICE_SVM_ATOMICS != 0,
            preffered_platform_atomic_alignment: svm_atomic_platform_align,
            preffered_platform_global_alignment: svm_atomic_global_align
        };
        let props = DeviceProps {
            compute_unit_count: cu_num,
            max_work_group_size: wg_max_size,
            max_alloc_size_in_bytes: max_alloc_size,
            global_mem_size: global_mem_size,
            shared_mem_caps: svm_caps,
            main_queue_is_async: false,
            supported_cl_version: cl_version
        };
        let dev_ext = DeviceSpecificExtData {
            handle: dev_han,
            command_queue: null_mut(),
            props: props
        };
        let dev = Device {
            ext: Box::new(dev_ext),
        };
        devs.push(dev);
    }

    let mut ret_c = 0;
    let props = null();
    let cl_ctx = clCreateContext(
        props,
        len,
        dev_ids.as_ptr(),
        None,
        null_mut(),
        &mut ret_c
    );
    if ret_c != cl_sys::CL_SUCCESS {
        match ret_c {
            cl_sys::CL_DEVICE_NOT_AVAILABLE => {
                panic!("One of devices is not available!")
            },
            cl_sys::CL_OUT_OF_RESOURCES |
            cl_sys::CL_OUT_OF_HOST_MEMORY => {
                return Err(OCLFailure::ResourcesExhausted);
            },
            cl_sys::CL_INVALID_PLATFORM |
            cl_sys::CL_INVALID_PROPERTY |
            cl_sys::CL_INVALID_DEVICE |
            _ => unreachable!()
        }
    }
    OCL_SHARED_CONTEXT.cl_contex = cl_ctx;
    for dev in &mut devs {
        let mut ret_code = CL_SUCCESS;
        let q = clCreateCommandQueue(
            cl_ctx,
            dev.ext.handle,
            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
            &mut ret_code
        );
        match ret_code {
            cl_sys::CL_SUCCESS => (),
            cl_sys::CL_OUT_OF_RESOURCES |
            cl_sys::CL_OUT_OF_HOST_MEMORY => {
                return Err(OCLFailure::ResourcesExhausted)
            },
            cl_sys::CL_INVALID_QUEUE_PROPERTIES |
            _ => unreachable!()
        }
        let mut cmd_q_props: cl_bitfield = 0;
        let ret_code = clGetCommandQueueInfo(
            q,
            CL_QUEUE_PROPERTIES,
            size_of::<cl_command_queue_properties>(),
            addr_of_mut!(cmd_q_props).cast(),
            null_mut()
        );
        match ret_code {
            cl_sys::CL_SUCCESS => (),
            cl_sys::CL_OUT_OF_RESOURCES |
            cl_sys::CL_OUT_OF_HOST_MEMORY => {
                return Err(OCLFailure::ResourcesExhausted)
            },
            _ => unreachable!()
        }
        let device_q_is_async = cmd_q_props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE != 0;
        dev.ext.props.main_queue_is_async = device_q_is_async;
        dev.ext.command_queue = q;
    }

    OCL_SHARED_CONTEXT.state.store(OCLSharedContextLifecycleState::DoneInit as _, Ordering::Release);

    return Ok(devs)
} }

#[test]
fn mem() {
    let devs = enumerate_devices().unwrap();

    let dev = &devs[0];

    let mut mem = dev.allocate_memory::<u32>(64).unwrap();

    let mut ix = 0;
    for item in mem.as_mut_items() {
        *item = ix;
        ix += 1;
    }
    let mut ix = 0;
    for item in mem.as_items() {
        // println!("{}", *item)
        assert!(*item == ix);
        ix += 1;
    }

    dev.deallocate_memory(mem);
}

#[test]
fn kernel_names() {

    #[allow(unused_variables)]
    let dev = enumerate_devices().unwrap();

    let text = "__kernel void lol() {}; __kernel void lol2() {}; __kernel void lol3() {}";

    let bundle = CodeBundle::from_textual_reprs(&[
        text.as_bytes()
    ]).unwrap();

    for name in bundle.get_available_kernel_names() {
        println!("{}", name)
    }
}
// #[test]
// fn ops_on_fd() {
//     let devs = enumerate_devices().unwrap();
//     let dev = &devs[0];

//     let item_count = 65535;
//     let mut mem = dev.allocate_memory::<u32>(item_count).unwrap();
//     let mut ix = 0;
//     for item in mem.as_mut_items() {
//         *item = ix;
//         ix += 1;
//     }

//     let text = r#"
//     __kernel void lol(__global uint* param1, uint param2) {
//         uint gix = get_global_id(0);
//         param1[gix] *= 2;
//     }"#;

//     let bundle = CodeBundle::from_textual_reprs(&[
//         text.as_bytes()
//     ]).unwrap();

//     let param = 2u32;
//     let kern = bundle.build_kernel("lol", (mem, param,)).unwrap();

//     let tok = dev.launch_kernel(kern, item_count, ).unwrap();

//     let fd = tok.as_fd().unwrap();

//     let mut evs = [(fd, libc::POLLIN, 0i16)];
//     loop {
//         let ret_code = unsafe { libc::poll(evs.as_mut_ptr().cast(), 1, -1) };
//         if ret_code == -1 {
//             unsafe { panic!("Failed {}", *libc::__errno_location()) }
//         }

//         let mut buf = [0u8;4];
//         unsafe { libc::read(fd, buf.as_mut_ptr().cast(), 4) };
//         let str = unsafe { core::str::from_utf8_unchecked(&buf) };
//         if str == "done" { break }
//     }

//     let mut ix = 0;
//     for i in mem.as_items() {
//         let k = ix * 2;
//         assert!(*i == k);
//         ix += 1;
//     }
// }

#[test]
fn ops_on_cb() {
    let devs = enumerate_devices().unwrap();
    let dev = &devs[0];

    let item_count = 65535;
    let mut mem = dev.allocate_memory::<u32>(item_count).unwrap();
    let mut ix = 0;
    for item in mem.as_mut_items() {
        *item = ix;
        ix += 1;
    }

    let text = r#"
    __kernel void lol(__global uint* param1, uint param2) {
        uint gix = get_global_id(0);
        param1[gix] *= 2;
    }"#;

    let bundle = CodeBundle::from_textual_reprs(&[
        text.as_bytes()
    ]).unwrap();

    let param = 2u32;
    let kern = bundle.build_kernel("lol", (mem, param,)).unwrap();

    let tok = dev.launch_kernel(kern, item_count, ).unwrap();

    let done = core::sync::atomic::AtomicBool::new(false);
    tok.attach_completion_callback(|_|{
        done.store(true, Ordering::Relaxed);
    }).unwrap();

    while !done.load(Ordering::Relaxed) {}

    // println!("{:#?}", mem.as_items());

    let mut ix = 0;
    for i in mem.as_items() {
        let k = ix * 2;
        assert!(*i == k);
        ix += 1;
    }
}

#[test]
fn wait_on_blocking_call() {
    // tok.await_completion().unwrap();
    let devs = enumerate_devices().unwrap();
    let dev = &devs[0];

    let item_count = 65535;
    let mut mem = dev.allocate_memory::<u32>(item_count).unwrap();
    let mut ix = 0;
    for item in mem.as_mut_items() {
        *item = ix;
        ix += 1;
    }

    let text = r#"
    __kernel void lol(__global uint* param1, uint param2) {
        uint gix = get_global_id(0);
        param1[gix] *= 2;
    }"#;

    let bundle = CodeBundle::from_textual_reprs(&[
        text.as_bytes()
    ]).unwrap();

    let param = 2u32;
    let kern = bundle.build_kernel("lol", (mem, param,)).unwrap();

    let tok = dev.launch_kernel(kern, item_count, ).unwrap();

    tok.await_completion().unwrap();

    // println!("{:#?}", mem.as_items());

    let mut ix = 0;
    for i in mem.as_items() {
        let k = ix * 2;
        assert!(*i == k);
        ix += 1;
    }
}

#[test]
fn wait_on_token() {

    let devs = enumerate_devices().unwrap();
    let dev = &devs[0];

    let item_count = 256;
    let mut mem = dev.allocate_memory::<u32>(item_count).unwrap();
    let mut ix = 0;
    for item in mem.as_mut_items() {
        *item = ix;
        ix += 1;
    }

    let text = r#"
    __kernel void lol(__global uint* param1, uint param2) {
        uint gix = get_global_id(0);
        param1[gix] *= 2;
    }"#;

    let bundle = CodeBundle::from_textual_reprs(&[
        text.as_bytes()
    ]).unwrap();
    let param = 2u32;
    let kern = bundle.build_kernel("lol", (mem, param,)).unwrap();
    let tok = dev.launch_kernel(kern, item_count, ).unwrap();

    let ft = tok.as_futex().unwrap();
    Token::await_completion_on_token_futex(ft);

    // println!("{:#?}", mem.as_items());

    let mut ix = 0;
    for i in mem.as_items() {
        let k = ix * 2;
        assert!(*i == k);
        ix += 1;
    }
}

#[test] #[ignore]
fn props() {
    let devs = enumerate_devices().unwrap();

    for dev in &devs {
        println!("{:#?}", dev.get_properties());
    }
}