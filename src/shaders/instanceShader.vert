#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;   // unused — color comes per-instance

layout(location = 2) in vec3 instancePos;
layout(location = 3) in vec4 instanceColor;

layout(location = 0) out vec4 fragColor;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition*0.3 + instancePos, 1.0);
    fragColor = instanceColor;
}
