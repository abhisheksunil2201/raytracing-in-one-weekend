const std = @import("std");
const math = std.math;

// constants
pub const infinity = std.math.inf(f64);
pub const pi = 3.1415926535897932385;

// util functions
pub fn degreesToRadians(degrees: f64) f64 {
    return degrees * pi / 180.0;
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
    const r = pixel_color.x();
    const g = pixel_color.y();
    const b = pixel_color.z();

    const rbyte = @as(u8, @intFromFloat(255.999 * r));
    const gbyte = @as(u8, @intFromFloat(255.999 * g));
    const bbyte = @as(u8, @intFromFloat(255.999 * b));
    try writer.print("{} {} {}\n", .{ rbyte, gbyte, bbyte });
}

pub fn ray_color(r: Ray, world: *const Hittable) Color {
    var rec: HitRecord = undefined;
    if (world.hit(r, Interval.initWithValues(0, infinity), &rec)) {
        const normal_color = Color.initWithValues(rec.normal.x() + 1, rec.normal.y() + 1, rec.normal.z() + 1);
        return Vec3.mulScalar(0.5, normal_color);
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
    pub const empty = Interval.init();
    pub const universe = Interval.initWithValues(-infinity, infinity);
};

pub fn main() !void {
    const aspect_ratio: f64 = 16.0 / 9.0;
    const image_width: u32 = 400;

    const temp_image_height: u32 = @intFromFloat(image_width / aspect_ratio);
    const image_height = if (temp_image_height < 1) 1 else temp_image_height;

    // World
    const allocator = std.heap.page_allocator;
    var world = HittableList.init(allocator);
    defer world.deinit();

    try world.add(Sphere.init(Point3.initWithValues(0, 0, -1), 0.5).toHittable());
    try world.add(Sphere.init(Point3.initWithValues(0, -100.5, -1), 100).toHittable());

    const hittable_world = world.toHittable();

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
            const pixel_color = ray_color(r, &hittable_world);

            try writeColor(file.writer(), pixel_color);
        }
    }
    std.debug.print("\rDone.\n", .{});
}
