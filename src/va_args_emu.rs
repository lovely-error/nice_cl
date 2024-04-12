use core::{any::TypeId, mem::{align_of_val, forget, size_of_val, transmute}, ptr::{addr_of, drop_in_place}};
#[repr(C)]
pub struct ErasedRef {
  pub data_ptr: *const u8,
  pub size: usize,
  pub alignment: usize,
  pub type_id: TypeId,
  pub dctor: fn(*mut u8)
}

pub trait KernelArgument: Sized {
    fn as_opaque(&self) -> ErasedRef;
}

macro_rules! autoimpl {
    ($trait:ident, $($conformer:ident),+) => {
      $(
        impl $trait for $conformer {
          fn as_opaque(&self) -> ErasedRef {
              ErasedRef {
                data_ptr: addr_of!(*self).cast(),
                size: size_of_val(self),
                alignment: align_of_val(self),
                type_id: TypeId::of::<Self>(),
                dctor: unsafe{transmute(drop_in_place::<Self> as *mut ())}
              }
          }
        }
      )+
    };
}

autoimpl!(
  KernelArgument,
  u8,
  u16,
  u32,
  u64,
  i8,
  i16,
  i32,
  i64
);

pub(crate) struct SomePointer {}
impl<T> KernelArgument for *mut T {
  fn as_opaque(&self) -> ErasedRef {
    ErasedRef {
      data_ptr: addr_of!(*self).cast(),
      size: size_of_val(self),
      alignment: align_of_val(self),
      type_id: TypeId::of::<SomePointer>(),
      dctor: unsafe{transmute(drop_in_place::<Self> as *mut ())}
    }
  }
}
impl KernelArgument for () {
  fn as_opaque(&self) -> ErasedRef {
    ErasedRef {
      data_ptr: addr_of!(*self).cast(),
      size: size_of_val(self),
      alignment: align_of_val(self),
      type_id: TypeId::of::<Self>(),
      dctor: unsafe{transmute(drop_in_place::<Self> as *mut ())}
    }
  }
}


pub trait KernelArguments {
    fn iter(self) -> impl Iterator<Item = ErasedRef>;
    fn len(&self) -> usize;
}

#[test] #[ignore = "this is for codegen"]
fn gen_stuff() {
  let limit = 16;

  let mut type_names = String::new();
  type_names.reserve(limit);
  let mut impls = String::new();
  impls.reserve(limit);
  let mut matchers = String::new();
  matchers.reserve(limit);

  let mut result = String::new();

  use core::fmt::Write;

  for ix in 0 .. limit {
    write!(&mut impls, "\nT{}:KernelArgument,", ix).unwrap();
    write!(&mut type_names, "T{},", ix).unwrap();
    write!(&mut matchers, "              {} => this.as_ref().unwrap().{}.as_opaque(),\n", ix, ix).unwrap();

    write!(&mut result,
      r"
impl<{}> KernelArguments for ({}) {{
  fn iter(self) -> impl Iterator<Item = OpaqueHandle> {{
      let mut ix = 0;
      let mut this = Some(self);
      core::iter::from_fn(move || {{
          let item = match ix {{
{}
              _ => {{
                  if let Some(this) = this.take() {{
                      forget(this)
                  }}
                  return None
              }}
          }};
          ix += 1;
          return Some(item)
      }})
  }}
  fn len(&self) -> usize {{
    {}
  }}
}}
      ",
      impls, type_names, matchers, ix + 1
    ).unwrap();
  }

  println!("{}", result)

}

impl KernelArguments for () {
  fn iter(self) -> impl Iterator<Item = ErasedRef> {
      core::iter::from_fn(|| None)
  }
  fn len(&self) -> usize {
      0
  }
}

impl<
T0:KernelArgument,> KernelArguments for (T0,) {
  fn iter(self) -> impl Iterator<Item = ErasedRef> {
      let mut ix = 0;
      let mut this = Some(self);
      core::iter::from_fn(move || {
          let item = match ix {
              0 => this.as_ref().unwrap().0.as_opaque(),

              _ => {
                  if let Some(this) = this.take() {
                      forget(this)
                  }
                  return None
              }
          };
          ix += 1;
          return Some(item)
      })
  }
  fn len(&self) -> usize {
    1
  }
}

impl<
T0:KernelArgument,
T1:KernelArgument,> KernelArguments for (T0,T1,) {
  fn iter(self) -> impl Iterator<Item = ErasedRef> {
      let mut ix = 0;
      let mut this = Some(self);
      core::iter::from_fn(move || {
          let item = match ix {
              0 => this.as_ref().unwrap().0.as_opaque(),
              1 => this.as_ref().unwrap().1.as_opaque(),

              _ => {
                  if let Some(this) = this.take() {
                      forget(this)
                  }
                  return None
              }
          };
          ix += 1;
          return Some(item)
      })
  }
  fn len(&self) -> usize {
    2
  }
}

impl<
T0:KernelArgument,
T1:KernelArgument,
T2:KernelArgument,> KernelArguments for (T0,T1,T2,) {
  fn iter(self) -> impl Iterator<Item = ErasedRef> {
      let mut ix = 0;
      let mut this = Some(self);
      core::iter::from_fn(move || {
          let item = match ix {
              0 => this.as_ref().unwrap().0.as_opaque(),
              1 => this.as_ref().unwrap().1.as_opaque(),
              2 => this.as_ref().unwrap().2.as_opaque(),

              _ => {
                  if let Some(this) = this.take() {
                      forget(this)
                  }
                  return None
              }
          };
          ix += 1;
          return Some(item)
      })
  }
  fn len(&self) -> usize {
    3
  }
}

impl<
T0:KernelArgument,
T1:KernelArgument,
T2:KernelArgument,
T3:KernelArgument,> KernelArguments for (T0,T1,T2,T3,) {
  fn iter(self) -> impl Iterator<Item = ErasedRef> {
      let mut ix = 0;
      let mut this = Some(self);
      core::iter::from_fn(move || {
          let item = match ix {
              0 => this.as_ref().unwrap().0.as_opaque(),
              1 => this.as_ref().unwrap().1.as_opaque(),
              2 => this.as_ref().unwrap().2.as_opaque(),
              3 => this.as_ref().unwrap().3.as_opaque(),

              _ => {
                  if let Some(this) = this.take() {
                      forget(this)
                  }
                  return None
              }
          };
          ix += 1;
          return Some(item)
      })
  }
  fn len(&self) -> usize {
    4
  }
}

impl<
T0:KernelArgument,
T1:KernelArgument,
T2:KernelArgument,
T3:KernelArgument,
T4:KernelArgument,> KernelArguments for (T0,T1,T2,T3,T4,) {
  fn iter(self) -> impl Iterator<Item = ErasedRef> {
      let mut ix = 0;
      let mut this = Some(self);
      core::iter::from_fn(move || {
          let item = match ix {
              0 => this.as_ref().unwrap().0.as_opaque(),
              1 => this.as_ref().unwrap().1.as_opaque(),
              2 => this.as_ref().unwrap().2.as_opaque(),
              3 => this.as_ref().unwrap().3.as_opaque(),
              4 => this.as_ref().unwrap().4.as_opaque(),

              _ => {
                  if let Some(this) = this.take() {
                      forget(this)
                  }
                  return None
              }
          };
          ix += 1;
          return Some(item)
      })
  }
  fn len(&self) -> usize {
    5
  }
}

impl<
T0:KernelArgument,
T1:KernelArgument,
T2:KernelArgument,
T3:KernelArgument,
T4:KernelArgument,
T5:KernelArgument,> KernelArguments for (T0,T1,T2,T3,T4,T5,) {
  fn iter(self) -> impl Iterator<Item = ErasedRef> {
      let mut ix = 0;
      let mut this = Some(self);
      core::iter::from_fn(move || {
          let item = match ix {
              0 => this.as_ref().unwrap().0.as_opaque(),
              1 => this.as_ref().unwrap().1.as_opaque(),
              2 => this.as_ref().unwrap().2.as_opaque(),
              3 => this.as_ref().unwrap().3.as_opaque(),
              4 => this.as_ref().unwrap().4.as_opaque(),
              5 => this.as_ref().unwrap().5.as_opaque(),

              _ => {
                  if let Some(this) = this.take() {
                      forget(this)
                  }
                  return None
              }
          };
          ix += 1;
          return Some(item)
      })
  }
  fn len(&self) -> usize {
    6
  }
}

impl<
T0:KernelArgument,
T1:KernelArgument,
T2:KernelArgument,
T3:KernelArgument,
T4:KernelArgument,
T5:KernelArgument,
T6:KernelArgument,> KernelArguments for (T0,T1,T2,T3,T4,T5,T6,) {
  fn iter(self) -> impl Iterator<Item = ErasedRef> {
      let mut ix = 0;
      let mut this = Some(self);
      core::iter::from_fn(move || {
          let item = match ix {
              0 => this.as_ref().unwrap().0.as_opaque(),
              1 => this.as_ref().unwrap().1.as_opaque(),
              2 => this.as_ref().unwrap().2.as_opaque(),
              3 => this.as_ref().unwrap().3.as_opaque(),
              4 => this.as_ref().unwrap().4.as_opaque(),
              5 => this.as_ref().unwrap().5.as_opaque(),
              6 => this.as_ref().unwrap().6.as_opaque(),

              _ => {
                  if let Some(this) = this.take() {
                      forget(this)
                  }
                  return None
              }
          };
          ix += 1;
          return Some(item)
      })
  }
  fn len(&self) -> usize {
    7
  }
}

impl<
T0:KernelArgument,
T1:KernelArgument,
T2:KernelArgument,
T3:KernelArgument,
T4:KernelArgument,
T5:KernelArgument,
T6:KernelArgument,
T7:KernelArgument,> KernelArguments for (T0,T1,T2,T3,T4,T5,T6,T7,) {
  fn iter(self) -> impl Iterator<Item = ErasedRef> {
      let mut ix = 0;
      let mut this = Some(self);
      core::iter::from_fn(move || {
          let item = match ix {
              0 => this.as_ref().unwrap().0.as_opaque(),
              1 => this.as_ref().unwrap().1.as_opaque(),
              2 => this.as_ref().unwrap().2.as_opaque(),
              3 => this.as_ref().unwrap().3.as_opaque(),
              4 => this.as_ref().unwrap().4.as_opaque(),
              5 => this.as_ref().unwrap().5.as_opaque(),
              6 => this.as_ref().unwrap().6.as_opaque(),
              7 => this.as_ref().unwrap().7.as_opaque(),

              _ => {
                  if let Some(this) = this.take() {
                      forget(this)
                  }
                  return None
              }
          };
          ix += 1;
          return Some(item)
      })
  }
  fn len(&self) -> usize {
    8
  }
}

impl<
T0:KernelArgument,
T1:KernelArgument,
T2:KernelArgument,
T3:KernelArgument,
T4:KernelArgument,
T5:KernelArgument,
T6:KernelArgument,
T7:KernelArgument,
T8:KernelArgument,> KernelArguments for (T0,T1,T2,T3,T4,T5,T6,T7,T8,) {
  fn iter(self) -> impl Iterator<Item = ErasedRef> {
      let mut ix = 0;
      let mut this = Some(self);
      core::iter::from_fn(move || {
          let item = match ix {
              0 => this.as_ref().unwrap().0.as_opaque(),
              1 => this.as_ref().unwrap().1.as_opaque(),
              2 => this.as_ref().unwrap().2.as_opaque(),
              3 => this.as_ref().unwrap().3.as_opaque(),
              4 => this.as_ref().unwrap().4.as_opaque(),
              5 => this.as_ref().unwrap().5.as_opaque(),
              6 => this.as_ref().unwrap().6.as_opaque(),
              7 => this.as_ref().unwrap().7.as_opaque(),
              8 => this.as_ref().unwrap().8.as_opaque(),

              _ => {
                  if let Some(this) = this.take() {
                      forget(this)
                  }
                  return None
              }
          };
          ix += 1;
          return Some(item)
      })
  }
  fn len(&self) -> usize {
    9
  }
}

impl<
T0:KernelArgument,
T1:KernelArgument,
T2:KernelArgument,
T3:KernelArgument,
T4:KernelArgument,
T5:KernelArgument,
T6:KernelArgument,
T7:KernelArgument,
T8:KernelArgument,
T9:KernelArgument,> KernelArguments for (T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,) {
  fn iter(self) -> impl Iterator<Item = ErasedRef> {
      let mut ix = 0;
      let mut this = Some(self);
      core::iter::from_fn(move || {
          let item = match ix {
              0 => this.as_ref().unwrap().0.as_opaque(),
              1 => this.as_ref().unwrap().1.as_opaque(),
              2 => this.as_ref().unwrap().2.as_opaque(),
              3 => this.as_ref().unwrap().3.as_opaque(),
              4 => this.as_ref().unwrap().4.as_opaque(),
              5 => this.as_ref().unwrap().5.as_opaque(),
              6 => this.as_ref().unwrap().6.as_opaque(),
              7 => this.as_ref().unwrap().7.as_opaque(),
              8 => this.as_ref().unwrap().8.as_opaque(),
              9 => this.as_ref().unwrap().9.as_opaque(),

              _ => {
                  if let Some(this) = this.take() {
                      forget(this)
                  }
                  return None
              }
          };
          ix += 1;
          return Some(item)
      })
  }
  fn len(&self) -> usize {
    10
  }
}

impl<
T0:KernelArgument,
T1:KernelArgument,
T2:KernelArgument,
T3:KernelArgument,
T4:KernelArgument,
T5:KernelArgument,
T6:KernelArgument,
T7:KernelArgument,
T8:KernelArgument,
T9:KernelArgument,
T10:KernelArgument,> KernelArguments for (T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,) {
  fn iter(self) -> impl Iterator<Item = ErasedRef> {
      let mut ix = 0;
      let mut this = Some(self);
      core::iter::from_fn(move || {
          let item = match ix {
              0 => this.as_ref().unwrap().0.as_opaque(),
              1 => this.as_ref().unwrap().1.as_opaque(),
              2 => this.as_ref().unwrap().2.as_opaque(),
              3 => this.as_ref().unwrap().3.as_opaque(),
              4 => this.as_ref().unwrap().4.as_opaque(),
              5 => this.as_ref().unwrap().5.as_opaque(),
              6 => this.as_ref().unwrap().6.as_opaque(),
              7 => this.as_ref().unwrap().7.as_opaque(),
              8 => this.as_ref().unwrap().8.as_opaque(),
              9 => this.as_ref().unwrap().9.as_opaque(),
              10 => this.as_ref().unwrap().10.as_opaque(),

              _ => {
                  if let Some(this) = this.take() {
                      forget(this)
                  }
                  return None
              }
          };
          ix += 1;
          return Some(item)
      })
  }
  fn len(&self) -> usize {
    11
  }
}

impl<
T0:KernelArgument,
T1:KernelArgument,
T2:KernelArgument,
T3:KernelArgument,
T4:KernelArgument,
T5:KernelArgument,
T6:KernelArgument,
T7:KernelArgument,
T8:KernelArgument,
T9:KernelArgument,
T10:KernelArgument,
T11:KernelArgument,> KernelArguments for (T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,) {
  fn iter(self) -> impl Iterator<Item = ErasedRef> {
      let mut ix = 0;
      let mut this = Some(self);
      core::iter::from_fn(move || {
          let item = match ix {
              0 => this.as_ref().unwrap().0.as_opaque(),
              1 => this.as_ref().unwrap().1.as_opaque(),
              2 => this.as_ref().unwrap().2.as_opaque(),
              3 => this.as_ref().unwrap().3.as_opaque(),
              4 => this.as_ref().unwrap().4.as_opaque(),
              5 => this.as_ref().unwrap().5.as_opaque(),
              6 => this.as_ref().unwrap().6.as_opaque(),
              7 => this.as_ref().unwrap().7.as_opaque(),
              8 => this.as_ref().unwrap().8.as_opaque(),
              9 => this.as_ref().unwrap().9.as_opaque(),
              10 => this.as_ref().unwrap().10.as_opaque(),
              11 => this.as_ref().unwrap().11.as_opaque(),

              _ => {
                  if let Some(this) = this.take() {
                      forget(this)
                  }
                  return None
              }
          };
          ix += 1;
          return Some(item)
      })
  }
  fn len(&self) -> usize {
    12
  }
}

impl<
T0:KernelArgument,
T1:KernelArgument,
T2:KernelArgument,
T3:KernelArgument,
T4:KernelArgument,
T5:KernelArgument,
T6:KernelArgument,
T7:KernelArgument,
T8:KernelArgument,
T9:KernelArgument,
T10:KernelArgument,
T11:KernelArgument,
T12:KernelArgument,> KernelArguments for (T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,) {
  fn iter(self) -> impl Iterator<Item = ErasedRef> {
      let mut ix = 0;
      let mut this = Some(self);
      core::iter::from_fn(move || {
          let item = match ix {
              0 => this.as_ref().unwrap().0.as_opaque(),
              1 => this.as_ref().unwrap().1.as_opaque(),
              2 => this.as_ref().unwrap().2.as_opaque(),
              3 => this.as_ref().unwrap().3.as_opaque(),
              4 => this.as_ref().unwrap().4.as_opaque(),
              5 => this.as_ref().unwrap().5.as_opaque(),
              6 => this.as_ref().unwrap().6.as_opaque(),
              7 => this.as_ref().unwrap().7.as_opaque(),
              8 => this.as_ref().unwrap().8.as_opaque(),
              9 => this.as_ref().unwrap().9.as_opaque(),
              10 => this.as_ref().unwrap().10.as_opaque(),
              11 => this.as_ref().unwrap().11.as_opaque(),
              12 => this.as_ref().unwrap().12.as_opaque(),

              _ => {
                  if let Some(this) = this.take() {
                      forget(this)
                  }
                  return None
              }
          };
          ix += 1;
          return Some(item)
      })
  }
  fn len(&self) -> usize {
    13
  }
}

impl<
T0:KernelArgument,
T1:KernelArgument,
T2:KernelArgument,
T3:KernelArgument,
T4:KernelArgument,
T5:KernelArgument,
T6:KernelArgument,
T7:KernelArgument,
T8:KernelArgument,
T9:KernelArgument,
T10:KernelArgument,
T11:KernelArgument,
T12:KernelArgument,
T13:KernelArgument,> KernelArguments for (T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,) {
  fn iter(self) -> impl Iterator<Item = ErasedRef> {
      let mut ix = 0;
      let mut this = Some(self);
      core::iter::from_fn(move || {
          let item = match ix {
              0 => this.as_ref().unwrap().0.as_opaque(),
              1 => this.as_ref().unwrap().1.as_opaque(),
              2 => this.as_ref().unwrap().2.as_opaque(),
              3 => this.as_ref().unwrap().3.as_opaque(),
              4 => this.as_ref().unwrap().4.as_opaque(),
              5 => this.as_ref().unwrap().5.as_opaque(),
              6 => this.as_ref().unwrap().6.as_opaque(),
              7 => this.as_ref().unwrap().7.as_opaque(),
              8 => this.as_ref().unwrap().8.as_opaque(),
              9 => this.as_ref().unwrap().9.as_opaque(),
              10 => this.as_ref().unwrap().10.as_opaque(),
              11 => this.as_ref().unwrap().11.as_opaque(),
              12 => this.as_ref().unwrap().12.as_opaque(),
              13 => this.as_ref().unwrap().13.as_opaque(),

              _ => {
                  if let Some(this) = this.take() {
                      forget(this)
                  }
                  return None
              }
          };
          ix += 1;
          return Some(item)
      })
  }
  fn len(&self) -> usize {
    14
  }
}

impl<
T0:KernelArgument,
T1:KernelArgument,
T2:KernelArgument,
T3:KernelArgument,
T4:KernelArgument,
T5:KernelArgument,
T6:KernelArgument,
T7:KernelArgument,
T8:KernelArgument,
T9:KernelArgument,
T10:KernelArgument,
T11:KernelArgument,
T12:KernelArgument,
T13:KernelArgument,
T14:KernelArgument,> KernelArguments for (T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,) {
  fn iter(self) -> impl Iterator<Item = ErasedRef> {
      let mut ix = 0;
      let mut this = Some(self);
      core::iter::from_fn(move || {
          let item = match ix {
              0 => this.as_ref().unwrap().0.as_opaque(),
              1 => this.as_ref().unwrap().1.as_opaque(),
              2 => this.as_ref().unwrap().2.as_opaque(),
              3 => this.as_ref().unwrap().3.as_opaque(),
              4 => this.as_ref().unwrap().4.as_opaque(),
              5 => this.as_ref().unwrap().5.as_opaque(),
              6 => this.as_ref().unwrap().6.as_opaque(),
              7 => this.as_ref().unwrap().7.as_opaque(),
              8 => this.as_ref().unwrap().8.as_opaque(),
              9 => this.as_ref().unwrap().9.as_opaque(),
              10 => this.as_ref().unwrap().10.as_opaque(),
              11 => this.as_ref().unwrap().11.as_opaque(),
              12 => this.as_ref().unwrap().12.as_opaque(),
              13 => this.as_ref().unwrap().13.as_opaque(),
              14 => this.as_ref().unwrap().14.as_opaque(),

              _ => {
                  if let Some(this) = this.take() {
                      forget(this)
                  }
                  return None
              }
          };
          ix += 1;
          return Some(item)
      })
  }
  fn len(&self) -> usize {
    15
  }
}

impl<
T0:KernelArgument,
T1:KernelArgument,
T2:KernelArgument,
T3:KernelArgument,
T4:KernelArgument,
T5:KernelArgument,
T6:KernelArgument,
T7:KernelArgument,
T8:KernelArgument,
T9:KernelArgument,
T10:KernelArgument,
T11:KernelArgument,
T12:KernelArgument,
T13:KernelArgument,
T14:KernelArgument,
T15:KernelArgument,> KernelArguments for (T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,) {
  fn iter(self) -> impl Iterator<Item = ErasedRef> {
      let mut ix = 0;
      let mut this = Some(self);
      core::iter::from_fn(move || {
          let item = match ix {
              0 => this.as_ref().unwrap().0.as_opaque(),
              1 => this.as_ref().unwrap().1.as_opaque(),
              2 => this.as_ref().unwrap().2.as_opaque(),
              3 => this.as_ref().unwrap().3.as_opaque(),
              4 => this.as_ref().unwrap().4.as_opaque(),
              5 => this.as_ref().unwrap().5.as_opaque(),
              6 => this.as_ref().unwrap().6.as_opaque(),
              7 => this.as_ref().unwrap().7.as_opaque(),
              8 => this.as_ref().unwrap().8.as_opaque(),
              9 => this.as_ref().unwrap().9.as_opaque(),
              10 => this.as_ref().unwrap().10.as_opaque(),
              11 => this.as_ref().unwrap().11.as_opaque(),
              12 => this.as_ref().unwrap().12.as_opaque(),
              13 => this.as_ref().unwrap().13.as_opaque(),
              14 => this.as_ref().unwrap().14.as_opaque(),
              15 => this.as_ref().unwrap().15.as_opaque(),

              _ => {
                  if let Some(this) = this.take() {
                      forget(this)
                  }
                  return None
              }
          };
          ix += 1;
          return Some(item)
      })
  }
  fn len(&self) -> usize {
    16
  }
}