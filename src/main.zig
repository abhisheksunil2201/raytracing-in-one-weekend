const std = @import("std");
const math = std.math;

// constants
pub const infinity = std.math.inf(f64);
pub const pi = 3.1415926535897932385;

// util functions
pub fn degreesToRadians(degrees: f64) f64 {
    return degrees * pi / 180.0;
}

pub fn randomDouble() f64 {
    // Returns random f64 in [0, 1)
    return std.crypto.random.float(f64);
}

pub fn randomDoubleRange(min: f64, max: f64) f64 {
    // Returns random f64 in [min, max)
    return min + (max - min) * randomDouble();
}

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

    pub fn random() Vec3 {
        return Vec3.initWithValues(randomDouble(), randomDouble(), randomDouble());
    }

    pub fn randomInRange(min: f64, max: f64) Vec3 {
        return Vec3.initWithValues(randomDoubleRange(min, max), randomDoubleRange(min, max), randomDoubleRange(min, max));
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

    pub fn randomUnitVector() Vec3 {
        while (true) {
            const p = Vec3.randomInRange(-1, 1);
            const lensq = p.lengthSquared();
            if (1e-160 < lensq and lensq <= 1) return Vec3.divScalar(p, math.sqrt(lensq));
        }
    }

    pub fn randomOnHemisphere(normal: Vec3) Vec3 {
        const onUnitSphere = randomUnitVector();
        if (dot(onUnitSphere, normal) > 0) {
            return onUnitSphere; // In the same hemisphere as normal
        } else {
            return Vec3.negate(onUnitSphere);
        }
    }

    pub fn linearToGamma(linear_component: f64) f64 {
        //to go from linear to gamma, we take inverse of gamma2
        //i.e. exponent of 1/gamma -> square root
        if (linear_component > 0) return math.sqrt(linear_component);
        return 0;
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

    pub fn print(v: Vec3) void {
        std.debug.print("{d} {d} {d}\n", .{ v.e[0], v.e[1], v.e[2] });
    }
};

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
        const scaled_dir = Vec3.mulScalar(t, self.dir);
        return Vec3.add(self.orig, scaled_dir);
    }
};

pub fn writeColor(writer: anytype, pixel_color: Color) !void {
    var r = pixel_color.x();
    var g = pixel_color.y();
    var b = pixel_color.z();

    //Apply linear to gamma transform for gamma 2
    r = Color.linearToGamma(r);
    g = Color.linearToGamma(g);
    b = Color.linearToGamma(b);

    const intensity = Interval.initWithValues(0.000, 0.999);
    const rbyte = @as(u8, @intFromFloat(256 * intensity.clamp(r)));
    const gbyte = @as(u8, @intFromFloat(256 * intensity.clamp(g)));
    const bbyte = @as(u8, @intFromFloat(256 * intensity.clamp(b)));
    try writer.print("{} {} {}\n", .{ rbyte, gbyte, bbyte });
}

pub fn ray_color(r: Ray, depth: i32, world: *const Hittable) Color {
    if (depth <= 0) return Color.init();

    var rec: HitRecord = undefined;
    if (world.hit(r, Interval.initWithValues(0.001, infinity), &rec)) {
        //Randomly generating vector using Lambertian distribution(more pronounced shadow)
        const direction = Vec3.add(rec.normal, Vec3.randomOnHemisphere(rec.normal));
        return Vec3.mulScalar(0.7, ray_color(Ray.initWithOriginAndDirection(rec.p, direction), depth - 1, world));
        //0.5 is the reflectance
    }

    const unit_direction = Vec3.unitVector(r.direction());
    const a = 0.5 * (unit_direction.y() + 1.0);
    const white = Color.initWithValues(1.0, 1.0, 1.0);
    const blue = Color.initWithValues(0.5, 0.7, 1.0);
    return Vec3.add(Vec3.mulScalar(1.0 - a, white), Vec3.mulScalar(a, blue));
}

pub const HitRecord = struct {
    p: Point3,
    normal: Vec3,
    t: f64,
    front_face: bool,

    pub fn set_face_normal(self: *HitRecord, r: Ray, outward_normal: Vec3) void {
        self.front_face = Vec3.dot(r.direction(), outward_normal) < 0;
        self.normal = if (self.front_face) outward_normal else outward_normal.negate();
    }
};

pub const Hittable = struct {
    const VTable = struct {
        hit: *const fn (self: *const anyopaque, r: Ray, ray_t: Interval, rec: *HitRecord) bool,
    };

    ptr: *const anyopaque,
    vtable: *const VTable,

    pub fn hit(self: Hittable, r: Ray, ray_t: Interval, rec: *HitRecord) bool {
        return self.vtable.hit(self.ptr, r, ray_t, rec);
    }
};

pub const Sphere = struct {
    center: Point3,
    radius: f64,

    pub fn init(center: Point3, radius: f64) Sphere {
        return Sphere{
            .center = center,
            .radius = if (radius > 0) radius else 0,
        };
    }

    pub fn toHittable(self: *const Sphere) Hittable {
        return Hittable{
            .ptr = self,
            .vtable = &.{
                .hit = hit,
            },
        };
    }

    fn hit(ctx: *const anyopaque, r: Ray, ray_t: Interval, rec: *HitRecord) bool {
        const self: *const Sphere = @ptrCast(@alignCast(ctx));
        const oc = Vec3.sub(self.center, r.origin());
        const a = Vec3.dot(r.direction(), r.direction());
        const h = Vec3.dot(r.direction(), oc);
        const c = Vec3.dot(oc, oc) - self.radius * self.radius;

        const discriminant = h * h - a * c;
        if (discriminant < 0) return false;

        const sqrtd = @sqrt(discriminant);
        // Find the nearest root that lies in the acceptable range
        var root = (h - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root)) {
                return false;
            }
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        const outward_normal = Vec3.divScalar(Vec3.sub(rec.p, self.center), self.radius);
        rec.set_face_normal(r, outward_normal);

        return true;
    }
};

pub const HittableList = struct {
    objects: std.ArrayList(Hittable),

    pub fn init(allocator: std.mem.Allocator) HittableList {
        return HittableList{
            .objects = std.ArrayList(Hittable).init(allocator),
        };
    }

    pub fn deinit(self: *HittableList) void {
        self.objects.deinit();
    }

    pub fn clear(self: *HittableList) void {
        self.objects.clearRetainingCapacity();
    }

    pub fn add(self: *HittableList, object: Hittable) !void {
        try self.objects.append(object);
    }

    pub fn hit(self: *const HittableList, r: Ray, ray_t: Interval, rec: *HitRecord) bool {
        var temp_rec: HitRecord = undefined;
        var hit_anything = false;
        var closest_so_far = ray_t.max;

        for (self.objects.items) |object| {
            if (object.hit(r, Interval.initWithValues(ray_t.min, closest_so_far), &temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec.* = temp_rec;
            }
        }

        return hit_anything;
    }

    pub fn toHittable(self: *const HittableList) Hittable {
        return Hittable{
            .ptr = self,
            .vtable = &.{
                .hit = hitImpl,
            },
        };
    }

    fn hitImpl(ctx: *const anyopaque, r: Ray, ray_t: Interval, rec: *HitRecord) bool {
        const self: *const HittableList = @ptrCast(@alignCast(ctx));
        return self.hit(r, ray_t, rec);
    }
};

pub fn hitSphere(center: Point3, radius: f64, ray: Ray) f64 {
    const oc = Vec3.sub(ray.origin(), center);
    const a = ray.direction().lengthSquared();
    const h = Vec3.dot(oc, ray.direction());
    const c = oc.lengthSquared() - radius * radius;
    const discriminant = h * h - a * c;
    if (discriminant < 0) {
        return -1;
    }
    return (h - math.sqrt(discriminant)) / a;
}

pub fn hitSphereColor(ray: Ray) Color {
    const t = hitSphere(Point3.initWithValues(0, 0, -1), 0.5, ray);
    if (t > 0) {
        const N = Vec3.unitVector(Vec3.sub(ray.at(t), Point3.initWithValues(0, 0, -1)));
        const color_with_normal = Color.initWithValues(N.x() + 1, N.y() + 1, N.z() + 1);
        return Vec3.mulScalar(0.5, color_with_normal);
    }

    const unit_direction = Vec3.unitVector(ray.direction());
    const a = 0.5 * (unit_direction.y() + 1.0);
    return Vec3.add(Vec3.mulScalar(1.0 - a, Color.initWithValues(1.0, 1.0, 1.0)), Vec3.mulScalar(a, Color.initWithValues(0.5, 0.7, 1.0)));
}

pub const Interval = struct {
    min: f64,
    max: f64,
    pub fn init() Interval {
        return Interval{
            .min = infinity,
            .max = -infinity,
        };
    }
    pub fn initWithValues(min_val: f64, max_val: f64) Interval {
        return Interval{
            .min = min_val,
            .max = max_val,
        };
    }
    pub fn size(self: Interval) f64 {
        return self.max - self.min;
    }
    pub fn contains(self: Interval, x: f64) bool {
        return self.min <= x and x <= self.max;
    }
    pub fn surrounds(self: Interval, x: f64) bool {
        return self.min < x and x < self.max;
    }
    pub fn clamp(self: Interval, x: f64) f64 {
        if (x < self.min) return self.min;
        if (x > self.max) return self.max;
        return x;
    }
    pub const empty = Interval.init();
    pub const universe = Interval.initWithValues(-infinity, infinity);
};

pub const Camera = struct {
    aspect_ratio: f64 = 1.0,
    image_width: u32 = 400,
    samples_per_pixel: u32 = 1.0, //Count of random samples for each pixel
    image_height: u32 = undefined,
    center: Point3 = undefined,
    pixel00_loc: Point3 = undefined,
    pixel_delta_u: Vec3 = undefined,
    pixel_delta_v: Vec3 = undefined,
    max_depth: i32 = 10,
    pub fn init() Camera {
        return Camera{
            .aspect_ratio = 1.0,
            .image_width = 100,
            .samples_per_pixel = 10,
            .image_height = undefined,
            .center = Point3.init(),
            .pixel00_loc = undefined,
            .pixel_delta_u = undefined,
            .pixel_delta_v = undefined,
        };
    }
    pub fn initialize(self: *Camera) void {
        self.image_height = @intFromFloat(@as(f64, @floatFromInt(self.image_width)) / self.aspect_ratio);
        self.image_height = if (self.image_height < 1) 1 else self.image_height;
        self.center = Point3.initWithValues(0, 0, 0);
        const focal_length: f64 = 1.0;
        const viewport_height: f64 = 2.0;
        const viewport_width: f64 = viewport_height * (@as(f64, @floatFromInt(self.image_width)) / @as(f64, @floatFromInt(self.image_height)));
        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        const viewport_u = Vec3.initWithValues(viewport_width, 0, 0);
        const viewport_v = Vec3.initWithValues(0, -viewport_height, 0);
        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        self.pixel_delta_u = Vec3.divScalar(viewport_u, @as(f64, @floatFromInt(self.image_width)));
        self.pixel_delta_v = Vec3.divScalar(viewport_v, @as(f64, @floatFromInt(self.image_height)));
        // Calculate the location of the upper left pixel.
        const viewport_upper_left = Vec3.sub(Vec3.sub(Vec3.sub(self.center, Vec3.initWithValues(0, 0, focal_length)), Vec3.divScalar(viewport_u, 2)), Vec3.divScalar(viewport_v, 2));
        self.pixel00_loc = Vec3.add(viewport_upper_left, Vec3.mulScalar(0.5, Vec3.add(self.pixel_delta_u, self.pixel_delta_v)));
    }
    fn sampleSquare() Vec3 {
        // Returns a random point in the [-0.5, -0.5]-[+0.5, +0.5] unit square
        return Vec3.initWithValues(randomDouble() - 0.5, // Using your existing random function
            randomDouble() - 0.5, 0);
    }
    pub fn getRay(self: *const Camera, i: u32, j: u32) Ray {
        // Get random offset within pixel
        const offset = sampleSquare();
        // Calculate sample position
        const pixel_sample = Vec3.add(Vec3.add(self.pixel00_loc, Vec3.mulScalar(@as(f64, @floatFromInt(i)) + offset.x(), self.pixel_delta_u)), Vec3.mulScalar(@as(f64, @floatFromInt(j)) + offset.y(), self.pixel_delta_v));
        const ray_origin = self.center;
        const ray_direction = Vec3.sub(pixel_sample, ray_origin);

        return Ray.initWithOriginAndDirection(ray_origin, ray_direction);
    }
    pub fn render(self: *Camera, world: *const Hittable, writer: anytype) !void {
        self.initialize();
        try writer.print("P3\n{} {}\n255\n", .{ self.image_width, self.image_height });

        const scale = 1.0 / @as(f64, @floatFromInt(self.samples_per_pixel));

        var j: u32 = 0;
        while (j < self.image_height) : (j += 1) {
            std.debug.print("\rScanlines remaining: {} ", .{self.image_height - j});
            var i: u32 = 0;
            while (i < self.image_width) : (i += 1) {
                var pixel_color = Color.init();
                var sample: u32 = 0;
                while (sample < self.samples_per_pixel) : (sample += 1) {
                    const r = self.getRay(i, j);
                    pixel_color.addAssign(ray_color(r, self.max_depth, world));
                }
                try writeColor(writer, Vec3.mulScalar(scale, pixel_color));
            }
        }
        std.debug.print("\rDone.\n", .{});
    }
};

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    //World
    var world = HittableList.init(allocator);
    defer world.deinit();

    try world.add(Sphere.init(Point3.initWithValues(0, 0, -1), 0.5).toHittable());
    try world.add(Sphere.init(Point3.initWithValues(0, -100.5, -1), 100).toHittable());
    const hittable_world = world.toHittable();
    var cam = Camera.init();
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth = 50;

    // Render
    const file = try std.fs.cwd().createFile("output.ppm", .{});
    defer file.close();
    try cam.render(&hittable_world, file.writer());
}
