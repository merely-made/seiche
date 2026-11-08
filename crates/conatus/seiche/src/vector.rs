//! The 2-vector the force code computes with. With the engine on it is rapier's
//! `Vector`; without, a plain pair with the operations the declarations and
//! the law parameters use, so none of that code needs rapier to compile.

#[cfg(feature = "engine")]
pub use rapier2d::prelude::Vector;

#[cfg(not(feature = "engine"))]
mod plain {
    use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Vector {
        pub x: f32,
        pub y: f32,
    }

    impl Vector {
        pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
        pub const fn new(x: f32, y: f32) -> Self {
            Self { x, y }
        }
        pub fn length_squared(self) -> f32 {
            self.x * self.x + self.y * self.y
        }
        pub fn length(self) -> f32 {
            self.length_squared().sqrt()
        }
        pub fn dot(self, o: Self) -> f32 {
            self.x * o.x + self.y * o.y
        }
        pub fn normalize_or_zero(self) -> Self {
            let l = self.length();
            if l > 0.0 { self / l } else { Self::ZERO }
        }
        pub fn distance(self, o: Self) -> f32 {
            (self - o).length()
        }
    }
    impl Add for Vector {
        type Output = Self;
        fn add(self, o: Self) -> Self {
            Self::new(self.x + o.x, self.y + o.y)
        }
    }
    impl Sub for Vector {
        type Output = Self;
        fn sub(self, o: Self) -> Self {
            Self::new(self.x - o.x, self.y - o.y)
        }
    }
    impl Mul<f32> for Vector {
        type Output = Self;
        fn mul(self, k: f32) -> Self {
            Self::new(self.x * k, self.y * k)
        }
    }
    impl Mul<Vector> for f32 {
        type Output = Vector;
        fn mul(self, v: Vector) -> Vector {
            v * self
        }
    }
    impl Div<f32> for Vector {
        type Output = Self;
        fn div(self, k: f32) -> Self {
            Self::new(self.x / k, self.y / k)
        }
    }
    impl Neg for Vector {
        type Output = Self;
        fn neg(self) -> Self {
            Self::new(-self.x, -self.y)
        }
    }
    impl AddAssign for Vector {
        fn add_assign(&mut self, o: Self) {
            *self = *self + o;
        }
    }
    impl SubAssign for Vector {
        fn sub_assign(&mut self, o: Self) {
            *self = *self - o;
        }
    }
}
#[cfg(not(feature = "engine"))]
pub use plain::Vector;
