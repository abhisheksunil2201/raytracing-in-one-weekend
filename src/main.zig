const std = @import("std");
const math = std.math;

pub const Vec3 = struct {
    e: [3]f64,

    // Constructor-like functions
    pub fn init() Vec3 {
        return Vec3{ .e = .{ 0, 0, 0 } };
    }

    pub fn initWithValues(e0: f64, e1: f64, e2: f64) Vec3 {
        return Vec3{ .e = .{ e0, e1, e2 } };
    }

    // Accessors
    pub fn x(self: Vec3) f64 {
        return self.e[0];
    }

    pub fn y(self: Vec3) f64 {
        return self.e[1];
    }

    pub fn z(self: Vec3) f64 {
        return self.e[2];
    }

    // Negation
    pub fn negate(self: Vec3) Vec3 {
        return Vec3{ .e = .{ -self.e[0], -self.e[1], -self.e[2] } };
    }

    // Indexing
    pub fn at(self: Vec3, index: usize) f64 {
        return self.e[index];
    }

    pub fn atMut(self: *Vec3, index: usize) *f64 {
        return &self.e[index];
    }

    // Compound assignment operators
    pub fn addAssign(self: *Vec3, other: Vec3) void {
        self.e[0] += other.e[0];
        self.e[1] += other.e[1];
        self.e[2] += other.e[2];
    }

    pub fn mulAssign(self: *Vec3, t: f64) void {
        self.e[0] *= t;
        self.e[1] *= t;
        self.e[2] *= t;
    }

    pub fn divAssign(self: *Vec3, t: f64) void {
        self.mulAssign(1 / t);
    }

    // Length and squared length
    pub fn length(self: Vec3) f64 {
        return math.sqrt(self.lengthSquared());
    }

    pub fn lengthSquared(self: Vec3) f64 {
        return self.e[0] * self.e[0] + self.e[1] * self.e[1] + self.e[2] * self.e[2];
    }
};

// Alias for point3
pub const Point3 = Vec3;

// Vector utility functions
pub fn add(u: Vec3, v: Vec3) Vec3 {
    return Vec3{ .e = .{ u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2] } };
}

pub fn sub(u: Vec3, v: Vec3) Vec3 {
    return Vec3{ .e = .{ u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2] } };
}

pub fn mulVec(u: Vec3, v: Vec3) Vec3 {
    return Vec3{ .e = .{ u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2] } };
}

pub fn mulScalar(t: f64, v: Vec3) Vec3 {
    return Vec3{ .e = .{ t * v.e[0], t * v.e[1], t * v.e[2] } };
}

pub fn divScalar(v: Vec3, t: f64) Vec3 {
    return mulScalar(1 / t, v);
}

pub fn dot(u: Vec3, v: Vec3) f64 {
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

pub fn cross(u: Vec3, v: Vec3) Vec3 {
    return Vec3{
        .e = .{
            u.e[1] * v.e[2] - u.e[2] * v.e[1],
            u.e[2] * v.e[0] - u.e[0] * v.e[2],
            u.e[0] * v.e[1] - u.e[1] * v.e[0],
        },
    };
}

pub fn unitVector(v: Vec3) Vec3 {
    return divScalar(v, v.length());
}

// Print function for Vec3
pub fn printVec3(v: Vec3) void {
    std.debug.print("{d} {d} {d}\n", .{ v.e[0], v.e[1], v.e[2] });
}

pub fn main() !void {
    const image_width = 256;
    const image_height = 256;
    // Open a file for writing
    const file = try std.fs.cwd().createFile("output.ppm", .{});
    defer file.close();
    //P3 Header
    try file.writer().print("P3\n{} {}\n255\n", .{ image_width, image_height });

    //Iterate over each pixel and output RGB
    var j: usize = 0;
    while (j < image_height) : (j += 1) {
        std.debug.print("\rScanlines remaining: {} ", .{image_height - j});
        var i: usize = 0;
        while (i < image_width) : (i += 1) {
            const r = @as(f64, @floatFromInt(i)) / @as(f64, image_width - 1);
            const g = @as(f64, @floatFromInt(j)) / @as(f64, image_height - 1);
            const b = 0.0;

            const ir: u8 = @intFromFloat(255.999 * r);
            const ig: u8 = @intFromFloat(255.999 * g);
            const ib: u8 = @intFromFloat(255.999 * b);

            try file.writer().print("{} {} {}\n", .{ ir, ig, ib });
        }
    }
    std.debug.print("\rDone.\n", .{});
}

pub fn writeColor(writer: anytype, pixel_color: Vec3) !void {
    const r = pixel_color.x();
    const g = pixel_color.y();
    const b = pixel_color.z();

    const rbyte = @as(u8, @intFromFloat(255.999 * r))
    const gbyte = @as(u8, @intFromFloat(255.999 * g))
    const bbyte = @as(u8, @intFromFloat(255.999 * b))

    try writer.print("{} {} {}\n", .{rbyte, gbyte, bbyte});
}
