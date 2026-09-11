// Particle render pass fragment shader.
// Renders each GL_POINT as a soft Gaussian circular glow disk.
// Combined with additive blending in Python, overlapping particles
// naturally accumulate into bright nebula clusters.

#version 430

in  vec3  v_color;
in  float v_alpha;
out vec4  fragColor;

void main() {
    // Distance from center of the GL_POINT square (0=center, 1=corner)
    vec2  coord = gl_PointCoord * 2.0 - 1.0;
    float dist  = length(coord);

    // Discard corners to form a clean circle
    if (dist > 1.0) discard;

    // Gaussian soft glow falloff
    float glow = exp(-dist * dist * 2.0);

    fragColor = vec4(v_color * (glow * 1.2), v_alpha * glow);
}
