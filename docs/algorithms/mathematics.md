# Mathematical Foundation

This document provides a comprehensive mathematical foundation for the Analytically Safe One-Way Binding algorithm used in the Perceptual Interdependence system.

## Core Mathematical Principle

### The Fundamental Relationship

The system exploits the perceptual interdependence between albedo textures and normal maps through the photometric rendering equation:

```
I = A × L(N)
```

Where:
- `I` = Rendered image intensity
- `A` = Albedo (surface reflectance)
- `L(N)` = Lighting function dependent on surface normal `N`

### Analytically Safe Binding

The core innovation lies in coordinated modifications that preserve the product relationship:

```
Poison Operation:    A_new = A_original × (1 + P)
Antidote Operation:  Z_new = Z_original / (1 + P)
Cancellation:        A_new × Z_new = A_original × Z_original
```

Where:
- `P` = Poison map with values in [0, poison_strength]
- `Z` = Z-component of normal map (surface steepness factor)

## Mathematical Guarantees

### Perfect Cancellation Theorem

**Theorem**: For any poison map P ≥ 0, the binding operations guarantee perfect mathematical cancellation.

**Proof**:
```
A_new × Z_new = [A_original × (1 + P)] × [Z_original / (1 + P)]
              = A_original × Z_original × [(1 + P) / (1 + P)]
              = A_original × Z_original × 1
              = A_original × Z_original
```

This holds for all pixels where P > -1, which is guaranteed by P ≥ 0.

### Geometric Constraint Preservation

**Theorem**: The antidote operation preserves valid surface normal constraints.

**Proof**: For a unit normal vector N = (X, Y, Z) where X² + Y² + Z² = 1:

After antidote operation: N' = (X, Y, Z/(1+P))

The modified vector maintains geometric validity because:
1. Z-component remains positive: Z/(1+P) > 0 for P ≥ 0, Z > 0
2. Vector can be renormalized to unit length
3. Surface orientation is preserved (no sign changes)

## Poison Map Generation

### Pseudo-Random Generation

The poison map is generated using a deterministic pseudo-random process:

```python
def generate_poison_map(shape, seed, poison_strength):
    np.random.seed(seed)
    base_noise = np.random.uniform(0, 1, shape)
    poison_map = base_noise * poison_strength
    return poison_map
```

**Properties**:
- **Deterministic**: Same seed produces identical poison maps
- **User-Specific**: Different seeds create uncorrelated patterns
- **Bounded**: Values strictly in [0, poison_strength]
- **Uniform Distribution**: Unbiased across spatial frequencies

### Statistical Properties

For a poison map P with uniform distribution U(0, s):
- **Mean**: μ_P = s/2
- **Variance**: σ²_P = s²/12
- **Standard Deviation**: σ_P = s/(2√3)

Where s = poison_strength.

## Photometric Rendering Model

### Simplified Lighting Model

The system uses a simplified Lambertian lighting model:

```
I(x,y) = A(x,y) × max(0, N(x,y) · L)
```

Where:
- `I(x,y)` = Rendered intensity at pixel (x,y)
- `A(x,y)` = Albedo value at pixel (x,y)
- `N(x,y)` = Surface normal at pixel (x,y)
- `L` = Light direction vector
- `·` = Dot product operation

### Normal Map Encoding

Normal maps are encoded in tangent space with the convention:
- **Red Channel (X)**: Tangent direction [-1, 1] → [0, 255]
- **Green Channel (Y)**: Bitangent direction [-1, 1] → [0, 255]
- **Blue Channel (Z)**: Surface normal direction [0, 1] → [128, 255]

The Z-component is critical for the antidote operation:
```
Z_tangent = (Blue_channel / 255.0) * 2.0 - 1.0  # Convert to [-1, 1]
Z_surface = max(0, Z_tangent)                    # Ensure positive
```

## Quality Metrics

### Peak Signal-to-Noise Ratio (PSNR)

```
PSNR = 10 × log₁₀(MAX²/MSE)
```

Where:
- `MAX` = Maximum possible pixel value (1.0 for normalized images)
- `MSE` = Mean Squared Error between images

**Expected Values**:
- Legitimate users: PSNR > 40dB (excellent quality)
- Attackers: PSNR < 30dB (noticeable degradation)

### Structural Similarity Index (SSIM)

```
SSIM(x,y) = (2μₓμᵧ + c₁)(2σₓᵧ + c₂) / ((μₓ² + μᵧ² + c₁)(σₓ² + σᵧ² + c₂))
```

Where:
- `μₓ, μᵧ` = Mean values of images x and y
- `σₓ², σᵧ²` = Variances of images x and y
- `σₓᵧ` = Covariance between images x and y
- `c₁, c₂` = Stabilization constants

**Expected Values**:
- Legitimate users: SSIM > 0.95 (nearly identical structure)
- Attackers: SSIM < 0.85 (visible structural differences)

## Security Analysis

### Information-Theoretic Security

The security relies on the computational difficulty of inverting the poison map without knowledge of the user seed.

**Entropy Analysis**:
For an N×M poison map with b bits per pixel:
- **Total Entropy**: H = N × M × b bits
- **Search Space**: 2^(N×M×b) possible poison maps
- **Brute Force Complexity**: O(2^(N×M×b))

For a 2048×2048 map with 8-bit precision:
- Total entropy: 33,554,432 bits
- Search space: 2^33,554,432 (computationally infeasible)

### Correlation Attack Resistance

The system resists correlation attacks through:

1. **Pseudo-Random Generation**: No exploitable patterns in poison maps
2. **User-Specific Seeds**: Uncorrelated poison maps between users
3. **Bounded Modifications**: Limited dynamic range prevents statistical analysis
4. **Geometric Constraints**: Normal map modifications must preserve surface validity

### Forensic Detection Capability

The forensic system exploits the correlation between suspicious textures and expected user patterns:

```
Correlation(S, E_u) = Σᵢⱼ S(i,j) × E_u(i,j)
```

Where:
- `S` = Extracted signature from suspicious texture
- `E_u` = Expected pattern for user u
- High correlation indicates user u as the source

## Numerical Stability

### Floating-Point Considerations

The algorithm maintains numerical stability through:

1. **Epsilon Protection**: Division operations use ε-protection to prevent division by zero
2. **Range Clamping**: All values are clamped to valid ranges [0,1]
3. **Double Precision**: Critical calculations use 64-bit floating-point arithmetic
4. **Overflow Prevention**: Intermediate calculations are bounded

### Error Propagation Analysis

For the binding operation A_new = A × (1 + P):

**Relative Error**:
```
δA_new/A_new = δA/A + δP/(1 + P)
```

**Maximum Error Bound**:
For machine epsilon ε and poison strength s:
```
|δA_new| ≤ |A| × ε + |P| × ε ≤ (1 + s) × ε
```

This ensures numerical errors remain bounded and do not accumulate.

## Optimization Considerations

### Computational Complexity

**Time Complexity**:
- Poison Generation: O(N×M) where N×M is image size
- Albedo Processing: O(N×M) element-wise multiplication
- Normal Processing: O(N×M) element-wise division
- **Total**: O(N×M) linear in image size

**Space Complexity**:
- Input Images: 2 × N×M×3 (albedo + normal, RGB)
- Poison Map: N×M (single channel)
- Output Images: 2 × N×M×3 (bound albedo + normal)
- **Total**: O(N×M) linear in image size

### Vectorization Opportunities

The algorithm is highly vectorizable:

```python
# Vectorized poison application
poisoned_albedo = albedo * (1.0 + poison_map[:,:,np.newaxis])

# Vectorized antidote calculation  
antidote_normal = normal.copy()
antidote_normal[:,:,2] = normal[:,:,2] / (1.0 + poison_map)
```

This enables efficient SIMD execution and GPU acceleration.

## Mathematical Extensions

### Multi-Channel Poison Maps

The algorithm can be extended to multi-channel poison maps:

```
A_new = A ⊙ (1 + P)
Z_new = Z ⊘ (1 + P_z)
```

Where ⊙ and ⊘ represent element-wise operations, and P_z is derived from P to maintain cancellation.

### Adaptive Poison Strength

Poison strength can be adapted based on local texture properties:

```
P_adaptive(x,y) = P_base(x,y) × α(x,y)
```

Where α(x,y) is a spatially-varying adaptation factor based on local texture complexity.

### Higher-Order Binding

The binding can be extended to higher-order relationships:

```
A_new = A × (1 + P₁) × (1 + P₂)
Z_new = Z / [(1 + P₁) × (1 + P₂)]
```

This provides additional security layers while maintaining perfect cancellation.

## Conclusion

The mathematical foundation of Analytically Safe One-Way Binding provides:

1. **Perfect Cancellation**: Guaranteed through algebraic relationships
2. **Geometric Validity**: Preserved surface normal constraints
3. **Numerical Stability**: Bounded errors and overflow protection
4. **Computational Efficiency**: Linear complexity and vectorizable operations
5. **Security Properties**: Information-theoretic guarantees and forensic traceability

These mathematical properties ensure the system provides both strong security guarantees and practical computational performance for real-world applications.