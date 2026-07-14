use crate::{IntCmp, IntWidth};
use anyhow::{Context, anyhow};
use llvm_sys_221::analysis::{LLVMVerifierFailureAction, LLVMVerifyModule};
use llvm_sys_221::core::{
    LLVMContextCreate, LLVMContextDispose, LLVMCreateBuilderInContext, LLVMDisposeBuilder,
    LLVMDisposeMessage, LLVMDisposeModule, LLVMInt1TypeInContext, LLVMInt8TypeInContext,
    LLVMInt16TypeInContext, LLVMInt32TypeInContext, LLVMInt64TypeInContext, LLVMIsConstant,
    LLVMIsNull, LLVMIsUndef, LLVMModuleCreateWithNameInContext, LLVMPointerTypeInContext,
};
use llvm_sys_221::error::{LLVMDisposeErrorMessage, LLVMGetErrorMessage};
use llvm_sys_221::orc2::lljit::{
    LLVMOrcCreateLLJIT, LLVMOrcCreateLLJITBuilder, LLVMOrcLLJITAddLLVMIRModuleWithRT,
    LLVMOrcLLJITGetMainJITDylib, LLVMOrcLLJITLookup,
};
use llvm_sys_221::orc2::{
    LLVMOrcCreateNewThreadSafeModule, LLVMOrcDisposeThreadSafeContext,
    LLVMOrcDisposeThreadSafeModule, LLVMOrcExecutorAddress, LLVMOrcJITDylibCreateResourceTracker,
    LLVMOrcReleaseResourceTracker, LLVMOrcResourceTrackerRef, LLVMOrcResourceTrackerRemove,
    LLVMOrcThreadSafeModuleRef,
};
use llvm_sys_221::prelude::LLVMTypeRef;
use llvm_sys_221::target::{
    LLVM_InitializeNativeAsmParser, LLVM_InitializeNativeAsmPrinter,
    LLVM_InitializeNativeDisassembler, LLVM_InitializeNativeTarget,
};
use llvm_sys_221::target_machine::{LLVMCodeGenOptLevel, LLVMGetDefaultTargetTriple};
use llvm_sys_221::{LLVMIntPredicate, LLVMValue, orc2};
use std::borrow::Cow;
use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::OnceLock;

pub(super) fn init_llvm() -> anyhow::Result<()> {
    #[derive(Debug, Copy, Clone)]
    enum InitError {
        NativeTarget,
        AsmParser,
        AsmPrinter,
        Disassembler,
    }

    static INIT_NATIVE: OnceLock<Result<(), InitError>> = OnceLock::new();

    let res = INIT_NATIVE.get_or_init(|| unsafe {
        if LLVM_InitializeNativeTarget() != 0 {
            return Err(InitError::NativeTarget);
        }

        if LLVM_InitializeNativeAsmParser() != 0 {
            return Err(InitError::AsmParser);
        }

        if LLVM_InitializeNativeAsmPrinter() != 0 {
            return Err(InitError::AsmPrinter);
        }

        if LLVM_InitializeNativeDisassembler() != 0 {
            return Err(InitError::Disassembler);
        }

        Ok(())
    });

    res.map_err(|err| anyhow!("LLVM native target init failed to init: {err:?}"))
}

// Worth noting that types seem to be singletons. At the very least, primitives are.
// Though this is likely only true per thread since LLVM claims to not be very thread-safe.
#[derive(PartialEq, Eq, Clone, Copy)]
pub(super) struct Type<'ctx> {
    ty: NonNull<llvm_sys_221::LLVMType>,
    _ctx: PhantomData<&'ctx LLVMContext>,
}

pub(super) fn int_cmp_to_predicate(cmp: IntCmp) -> LLVMIntPredicate {
    match cmp {
        IntCmp::Equal => LLVMIntPredicate::LLVMIntEQ,
        IntCmp::NotEqual => LLVMIntPredicate::LLVMIntNE,
        IntCmp::SignedLessThan => LLVMIntPredicate::LLVMIntSLT,
        IntCmp::SignedLessThanOrEqual => LLVMIntPredicate::LLVMIntSLE,
        IntCmp::SignedGreaterThan => LLVMIntPredicate::LLVMIntSGT,
        IntCmp::SignedGreaterThanOrEqual => LLVMIntPredicate::LLVMIntSGE,
        IntCmp::UnsignedLessThan => LLVMIntPredicate::LLVMIntULT,
        IntCmp::UnsignedLessThanOrEqual => LLVMIntPredicate::LLVMIntULE,
        IntCmp::UnsignedGreaterThan => LLVMIntPredicate::LLVMIntULT,
        IntCmp::UnsignedGreaterThanOrEqual => LLVMIntPredicate::LLVMIntULE,
    }
}

impl Type<'_> {
    pub fn new<'a>(ty_ref: LLVMTypeRef) -> Type<'a> {
        let ty = NonNull::new(ty_ref).unwrap();
        Type {
            ty,
            _ctx: PhantomData,
        }
    }
}

pub(super) fn int_width_to_llvm(ctx: &LLVMContext, width: IntWidth) -> Type<'_> {
    match width {
        IntWidth::W8 => unsafe { Type::new(LLVMInt8TypeInContext(ctx.0.as_ptr())) },
        IntWidth::W16 => unsafe { Type::new(LLVMInt16TypeInContext(ctx.0.as_ptr())) },
        IntWidth::W32 => unsafe { Type::new(LLVMInt32TypeInContext(ctx.0.as_ptr())) },
        IntWidth::W64 => unsafe { Type::new(LLVMInt64TypeInContext(ctx.0.as_ptr())) },
    }
}

pub(super) fn ir_type_to_llvm(ctx: &LLVMContext, ty: crate::Type) -> Type<'_> {
    match ty {
        crate::Type::Bool => unsafe { Type::new(LLVMInt1TypeInContext(ctx.0.as_ptr())) },
        crate::Type::Int(w) => int_width_to_llvm(ctx, w),
        crate::Type::HostPtr => unsafe {
            // https://llvm-swift.github.io/LLVMSwift/Structs/AddressSpace.html#/s:4LLVM12AddressSpaceV4zeroACvpZ
            let address_space = 0;
            Type::new(LLVMPointerTypeInContext(ctx.0.as_ptr(), address_space))
        },
    }
}

pub(super) struct LLVMString(NonNull<std::ffi::c_char>);

impl LLVMString {
    pub unsafe fn from_ptr(ptr: NonNull<std::ffi::c_char>) -> Self {
        Self(ptr)
    }

    pub fn as_cstr(&self) -> &CStr {
        unsafe { CStr::from_ptr(self.0.as_ptr()) }
    }
}

unsafe impl Send for LLVMString {}
unsafe impl Sync for LLVMString {}

impl Drop for LLVMString {
    fn drop(&mut self) {
        unsafe { LLVMDisposeMessage(self.0.as_ptr()) }
    }
}

pub(super) struct TargetTriple(LLVMString);

impl TargetTriple {
    pub fn get_default_triple() -> anyhow::Result<Self> {
        let string = NonNull::new(unsafe { LLVMGetDefaultTargetTriple() })
            .context("failed to get default target triple")?;

        let str = unsafe { LLVMString::from_ptr(string) };

        Ok(Self(str))
    }
}

pub(super) struct LLVMContext(NonNull<llvm_sys_221::LLVMContext>);

impl LLVMContext {
    /// Creates a new `Context`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use inkwell::context::Context;
    ///
    /// let context = Context::create();
    /// ```
    pub fn create() -> anyhow::Result<Self> {
        let context = NonNull::new(unsafe { LLVMContextCreate() })
            .context("failed to create llvm context")?;

        Ok(Self(context))
    }
}

unsafe impl Send for LLVMContext {}

impl Drop for LLVMContext {
    fn drop(&mut self) {
        unsafe { LLVMContextDispose(self.0.as_ptr()) }
    }
}

struct RawModule(NonNull<llvm_sys_221::LLVMModule>);

impl Drop for RawModule {
    fn drop(&mut self) {
        unsafe { LLVMDisposeModule(self.0.as_ptr()) }
    }
}

pub(super) struct Module<'ctx> {
    module: RawModule,
    _ctx: PhantomData<&'ctx LLVMContext>,
}

impl<'ctx> Module<'ctx> {
    pub fn validate(&self) -> anyhow::Result<()> {
        unsafe {
            let mut err_msg = std::ptr::null_mut();
            let failed = LLVMVerifyModule(
                self.module.0.as_ptr(),
                LLVMVerifierFailureAction::LLVMReturnStatusAction,
                &mut err_msg,
            );
            if failed != 0 {
                let s = CStr::from_ptr(err_msg).to_string_lossy().into_owned();
                LLVMDisposeMessage(err_msg);
                anyhow::bail!("Module validation failed: {s}");
            }
            // LLVMVerifyModule always allocates err_msg, free it even on success
            if !err_msg.is_null() {
                LLVMDisposeMessage(err_msg);
            }
            Ok(())
        }
    }

    pub unsafe fn into_raw(self) -> RawModule {
        self.module
    }
}

pub(super) struct Builder<'ctx> {
    builder: NonNull<llvm_sys_221::LLVMBuilder>,
    _ctx: PhantomData<&'ctx LLVMContext>,
}

impl Drop for Builder<'_> {
    fn drop(&mut self) {
        unsafe { LLVMDisposeBuilder(self.builder.as_ptr()) }
    }
}

fn to_cstr(str: &str) -> Cow<'_, CStr> {
    match str.as_bytes().contains(&b'\0') {
        true => unsafe { Cow::Borrowed(CStr::from_ptr(str.as_ptr().cast())) },
        false => {
            let mut vec = Vec::with_capacity(str.len().strict_add(1));
            vec.push(b'\0');
            unsafe { Cow::Owned(CString::from_vec_with_nul_unchecked(vec)) }
        }
    }
}

impl LLVMContext {
    pub fn create_module(&self, name: &str) -> anyhow::Result<Module<'_>> {
        let cstr = to_cstr(name);
        let builder = unsafe { LLVMModuleCreateWithNameInContext(cstr.as_ptr(), self.0.as_ptr()) };

        let module = NonNull::new(builder).context("failed to create llvm builder")?;

        Ok(Module {
            module: RawModule(module),
            _ctx: PhantomData,
        })
    }

    pub fn create_builder(&self) -> anyhow::Result<Builder<'_>> {
        let builder = unsafe { LLVMCreateBuilderInContext(self.0.as_ptr()) };
        let builder = NonNull::new(builder).context("failed to create llvm builder")?;

        Ok(Builder {
            builder,
            _ctx: PhantomData,
        })
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub(super) enum OptimizationLevel {
    None = 0,
    Less = 1,
    Default = 2,
    Aggressive = 3,
}

impl From<OptimizationLevel> for LLVMCodeGenOptLevel {
    fn from(value: OptimizationLevel) -> Self {
        match value {
            OptimizationLevel::None => LLVMCodeGenOptLevel::LLVMCodeGenLevelNone,
            OptimizationLevel::Less => LLVMCodeGenOptLevel::LLVMCodeGenLevelLess,
            OptimizationLevel::Default => LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault,
            OptimizationLevel::Aggressive => LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
        }
    }
}

pub(super) struct LLJit(NonNull<orc2::lljit::LLVMOrcOpaqueLLJIT>);

unsafe impl Send for LLJit {}
unsafe impl Sync for LLJit {}

impl LLJit {
    pub fn new() -> anyhow::Result<Self> {
        unsafe {
            let builder = LLVMOrcCreateLLJITBuilder();
            let mut jit = std::ptr::null_mut();
            let err = LLVMOrcCreateLLJIT(&mut jit, builder);
            // builder is consumed by LLVMOrcCreateLLJIT regardless of success/failure
            if !err.is_null() {
                let msg = LLVMGetErrorMessage(err);
                let s = CStr::from_ptr(msg).to_string_lossy().into_owned();
                LLVMDisposeErrorMessage(msg);
                anyhow::bail!("Failed to create LLJIT: {s}");
            }

            let jit =
                NonNull::new(jit).context("LLVMOrcCreateLLJIT returned null without error")?;

            Ok(Self(jit))
        }
    }

    /// # Safety
    ///
    /// `module` must be owned by the ctx
    pub(super) unsafe fn add_dynarec_block(
        &self,
        function_name: &str,
        ctx: LLVMContext,
        module: RawModule,
    ) -> anyhow::Result<(NonNull<()>, impl Drop + Send + Sync)> {
        // look up the compiled function
        let name = CString::new(function_name).context("function name contained null")?;

        unsafe {
            let tsc = orc2::LLVMOrcCreateNewThreadSafeContextFromLLVMContext(ctx.0.as_ptr());
            std::mem::forget(ctx);

            struct TSM(LLVMOrcThreadSafeModuleRef);

            impl Drop for TSM {
                fn drop(&mut self) {
                    unsafe { LLVMOrcDisposeThreadSafeModule(self.0) }
                }
            }

            let tsm = LLVMOrcCreateNewThreadSafeModule(module.0.as_ptr(), tsc);
            std::mem::forget(module);

            let tsm = TSM(tsm);

            LLVMOrcDisposeThreadSafeContext(tsc);

            // create a resource tracker so we can free this module individually
            let dylib = LLVMOrcLLJITGetMainJITDylib(self.0.as_ptr());
            let rt = LLVMOrcJITDylibCreateResourceTracker(dylib);

            let err = LLVMOrcLLJITAddLLVMIRModuleWithRT(self.0.as_ptr(), rt, tsm.0);
            std::mem::forget(tsm);
            if !err.is_null() {
                LLVMOrcReleaseResourceTracker(rt);
                let msg = LLVMGetErrorMessage(err);
                let s = CStr::from_ptr(msg).to_string_lossy().into_owned();
                LLVMDisposeErrorMessage(msg);
                anyhow::bail!("Failed to add IR module: {s}");
            }

            struct JitModule(LLVMOrcResourceTrackerRef);

            unsafe impl Send for JitModule {}
            unsafe impl Sync for JitModule {}

            impl Drop for JitModule {
                fn drop(&mut self) {
                    unsafe {
                        let err = LLVMOrcResourceTrackerRemove(self.0);
                        LLVMOrcReleaseResourceTracker(self.0);
                        if !err.is_null() {
                            let msg = LLVMGetErrorMessage(err);
                            let s = CStr::from_ptr(msg).to_string_lossy().into_owned();
                            LLVMDisposeErrorMessage(msg);
                            panic!("Failed to remove resource tracker: {s}");
                        }
                    }
                }
            }

            let module = JitModule(rt);

            let mut addr: LLVMOrcExecutorAddress = 0;
            let err = LLVMOrcLLJITLookup(self.0.as_ptr(), &mut addr, name.as_ptr());
            if !err.is_null() {
                LLVMOrcReleaseResourceTracker(rt);
                let msg = LLVMGetErrorMessage(err);
                let s = CStr::from_ptr(msg).to_string_lossy().into_owned();
                LLVMDisposeErrorMessage(msg);
                anyhow::bail!("Failed to look up symbol '{}': {s}", function_name);
            }

            let addr = usize::try_from(addr).unwrap_unchecked();
            let ptr = std::ptr::without_provenance::<()>(addr);
            let ptr = NonNull::new(ptr.cast_mut()).context("LLVM returned null ptr")?;

            Ok((ptr, module))
        }
    }
}

impl Drop for LLJit {
    fn drop(&mut self) {
        unsafe {
            let err = orc2::lljit::LLVMOrcDisposeLLJIT(self.0.as_ptr());
            if !err.is_null() {
                // both extracts the message and consumes the error in one shot
                let msg = LLVMGetErrorMessage(err);
                let s = CStr::from_ptr(msg).to_string_lossy().into_owned();
                LLVMDisposeErrorMessage(msg);
                panic!("Failed to release LLJIT context: {s}")
            }
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct Value<'ctx> {
    value: NonNull<LLVMValue>,
    _ctx: PhantomData<&'ctx LLVMContext>,
}

impl Value<'_> {
    fn is_undef(self) -> bool {
        unsafe { LLVMIsUndef(self.value.as_ptr()) == 1 }
    }

    fn is_null(self) -> bool {
        unsafe { LLVMIsNull(self.value.as_ptr()) == 1 }
    }

    fn is_const(self) -> bool {
        unsafe { LLVMIsConstant(self.value.as_ptr()) == 1 }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct FunctionValue<'ctx>(Value<'ctx>);

impl<'ctx> FunctionValue<'ctx> {}
