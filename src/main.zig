const std = @import("std");
const math = std.math;

pub const Color = Vec3;
pub const Point3 = Vec3;

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

    pub fn unitVector(v: Vec3) Vec3 {
        return divScalar(v, v.length());
    }
};

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
    return Vec3.divScalar(v, v.length());
}

// Print function for Vec3
pub fn printVec3(v: Vec3) void {
    std.debug.print("{d} {d} {d}\n", .{ v.e[0], v.e[1], v.e[2] });
}

pub const Ray = struct {
    orig: Point3,
    dir: Vec3,

    pub fn init() Ray {
        return Ray{ .orig = Point3.init(), .dir = Vec3.init() };
    }

    pub fn initWithOriginAndDirection(origin_param: Point3, direction_param: Vec3) Ray {
        return Ray{ .orig = origin_param, .dir = direction_param };
    }

    pub fn origin(self: Ray) Point3 {
        return self.orig;
    }

    pub fn direction(self: Ray) Vec3 {
        return self.dir;
    }

    pub fn at(self: Ray, t: f64) Point3 {
        const scaled_dir = Vec3.mutScalar(t, self.dir);
        return Vec3.add(self.orig, scaled_dir);
    }
};

pub fn writeColor(writer: anytype, pixel_color: Color) !void {
    const r = pixel_color.x();
    const g = pixel_color.y();
    const b = pixel_color.z();

    const rbyte = @as(u8, @intFromFloat(255.999 * r));
    const gbyte = @as(u8, @intFromFloat(255.999 * g));
    const bbyte = @as(u8, @intFromFloat(255.999 * b));
    try writer.print("{} {} {}\n", .{ rbyte, gbyte, bbyte });
}

pub fn ray_color(r: Ray) Vec3 {
    const unit_direction = Vec3.unitVector(r.direction());
    const a = 0.5 * (unit_direction.y() + 1.0);
    const white = Color.initWithValues(1.0, 1.0, 1.0);
    const blue = Color.initWithValues(0.5, 0.7, 1.0);
    return Vec3.add(Vec3.mulScalar(1.0 - a, white), Vec3.mulScalar(a, blue));
}

pub fn main() !void {
    const aspect_ratio: f64 = 16.0 / 9.0;
    const image_width: u32 = 400;

    const temp_image_height: u32 = @intFromFloat(image_width / aspect_ratio);
    const image_height = if (temp_image_height < 1) 1 else temp_image_height;

    // Camera

    const focal_length: f64 = 1.0;
    const viewport_height: f64 = 2.0;
    const viewport_width: f64 = viewport_height * (@as(f64, @floatFromInt(image_width)) / @as(f64, @floatFromInt(image_height)));
    const camera_center = Point3.initWithValues(0, 0, 0);

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    const viewport_u = Vec3.initWithValues(viewport_width, 0, 0);
    const viewport_v = Vec3.initWithValues(0, -viewport_height, 0);

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    const pixel_delta_u = Vec3.divScalar(viewport_u, @as(f64, @floatFromInt(image_width)));
    const pixel_delta_v = Vec3.divScalar(viewport_v, @as(f64, @floatFromInt(image_height)));

    // Calculate the location of the upper left pixel.
    const viewport_upper_left = Vec3.sub(Vec3.sub(Vec3.sub(camera_center, Vec3.initWithValues(0, 0, focal_length)), Vec3.divScalar(viewport_u, 2)), Vec3.divScalar(viewport_v, 2));
    const pixel00_loc = Vec3.add(viewport_upper_left, Vec3.mulScalar(0.5, Vec3.add(pixel_delta_u, pixel_delta_v)));

    // Open a file for writing
    const file = try std.fs.cwd().createFile("output.ppm", .{});
    defer file.close();
    //P3 Header
    try file.writer().print("P3\n{} {}\n255\n", .{ image_width, image_height });

    //Iterate over each pixel and output RGB
    var j: usize = 0;
    while (j < image_height) : (j += 1) {
        std.debug.print("\rScanlines remaining: {} ", .{image_height - j});
        var i: u32 = 0;
        while (i < image_width) : (i += 1) {
            const pixel_center = Vec3.add(Vec3.add(pixel00_loc, Vec3.mulScalar(@floatFromInt(i), pixel_delta_u)), Vec3.mulScalar(@floatFromInt(j), pixel_delta_v));
            const ray_direction = Vec3.sub(pixel_center, camera_center);
            const r = Ray.initWithOriginAndDirection(camera_center, ray_direction);
            const pixel_color = ray_color(r);

            try writeColor(file.writer(), pixel_color);
        }
    }
    std.debug.print("\rDone.\n", .{});
}
