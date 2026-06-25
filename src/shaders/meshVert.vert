#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 viewPos; // CRITICO: Passa la posizione della telecamera da CPU per i riflessi
} ubo;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec4 fragColor;

void main() {
    // Posizione finale sullo schermo
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPos, 1.0);

    // Trasformazione delle coordinate nello spazio del mondo (World Space)
    vec3 worldPos = vec3(ubo.model * vec4(inPos, 1.0));
    vec3 N = normalize(mat3(transpose(inverse(ubo.model))) * inNormal);
    
    // Direzione della luce (Sole) e della vista (Telecamera)
    vec3 L = normalize(vec3(1.0, 3.0, 1.0)); // Luce inclinata dall'alto
    vec3 V = normalize(ubo.viewPos - worldPos);

    
    vec3 waterColor = vec3(0.005, 0.08, 0.22);

    // 2. Luce Diffusa (Illuminazione generale del fluido)
    float diff = max(dot(N, L), 0.0);
    vec3 diffuseLight = diff * vec3(0.8, 0.9, 1.0); // Luce solare leggermente fredda

    // 3. Luce Speculare / Blinn-Phong (La lucentezza del sole sull'acqua)
    vec3 H = normalize(L + V);
    // Un esponente alto (128.0) crea punti di luce piccoli e taglienti, tipici dei liquidi
    float spec = pow(max(dot(N, H), 0.0), 128.0); 
    vec3 specularLight = spec * vec3(1.0, 1.0, 1.0) * 1.5; // Sole bianco e intenso

    // 4. Effetto Fresnel (L'acqua è più riflettente se guardata radente)
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 5.0); // Esponente più alto = transizione più netta
    fresnel = clamp(fresnel, 0.0, 1);

    vec3 finalColor = (0.1 + diffuseLight * 0.15) * waterColor + specularLight + (fresnel * vec3(0.1, 0.4, 0.6));

    fragColor = vec4(finalColor, 0.8);
}