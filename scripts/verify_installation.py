#!/usr/bin/env python3
"""
Installation verification script for perceptual interdependence package.
"""

import sys
import traceback
from pathlib import Path


def test_imports():
    """Test that all core modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from perceptual_interdependence import AssetBinder, CPUOptimizedMath
        print("  ✅ Core modules imported successfully")
        
        from perceptual_interdependence.utils.validation import ValidationSuite
        print("  ✅ Validation module imported successfully")
        
        from perceptual_interdependence.utils.texture_processing import TextureProcessor
        print("  ✅ Texture processing module imported successfully")
        
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_basic_functionality():
    """Test basic functionality."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        from perceptual_interdependence.algorithms.cpu_math import get_cpu_math
        
        cpu_math = get_cpu_math()
        print(f"  ✅ CPU Math initialized (Numba: {cpu_math.use_numba})")
        
        # Test poison map generation
        poison_map = cpu_math.generate_poison_map((64, 64), 42, 0.2)
        print(f"  ✅ Poison map generated: {poison_map.shape}")
        
        # Test validation
        from perceptual_interdependence.utils.validation import ValidationSuite
        validator = ValidationSuite()
        results = validator.validate_system_integrity()
        
        if results['valid']:
            print("  ✅ System integrity validation passed")
        else:
            print(f"  ⚠️  System integrity issues: {results['issues']}")
        
        return True
    except Exception as e:
        print(f"  ❌ Functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_performance():
    """Test performance benchmarking."""
    print("\n⚡ Testing performance...")
    
    try:
        from perceptual_interdependence.algorithms.cpu_math import get_cpu_math
        
        cpu_math = get_cpu_math()
        results = cpu_math.benchmark_performance((512, 512))
        
        total_time = results['total']
        throughput = (512 * 512) / total_time / 1e6  # Mpixels/sec
        
        print(f"  ✅ Performance test completed")
        print(f"     Processing time: {total_time:.3f}s")
        print(f"     Throughput: {throughput:.1f} Mpixels/sec")
        
        if total_time < 1.0:  # Should be fast
            print("  ✅ Performance is acceptable")
        else:
            print("  ⚠️  Performance is slower than expected")
        
        return True
    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
        traceback.print_exc()
        return False


def test_cli():
    """Test CLI availability."""
    print("\n🖥️  Testing CLI...")
    
    try:
        from perceptual_interdependence.cli.main import create_parser
        
        parser = create_parser()
        print("  ✅ CLI parser created successfully")
        
        # Test that help works
        help_text = parser.format_help()
        if "Perceptual Interdependence" in help_text:
            print("  ✅ CLI help text generated")
        else:
            print("  ⚠️  CLI help text incomplete")
        
        return True
    except Exception as e:
        print(f"  ❌ CLI test failed: {e}")
        traceback.print_exc()
        return False


def check_dependencies():
    """Check that all required dependencies are available."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('numba', 'Numba'),
        ('matplotlib', 'Matplotlib'),
    ]
    
    optional_packages = [
        ('streamlit', 'Streamlit'),
        ('pytest', 'Pytest'),
    ]
    
    all_good = True
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {name} available")
        except ImportError:
            print(f"  ❌ {name} missing (required)")
            all_good = False
    
    for package, name in optional_packages:
        try:
            __import__(package)
            print(f"  ✅ {name} available (optional)")
        except ImportError:
            print(f"  ⚠️  {name} missing (optional)")
    
    return all_good


def main():
    """Run all verification tests."""
    print("🔬 Perceptual Interdependence Installation Verification")
    print("=" * 60)
    
    tests = [
        ("Dependencies", check_dependencies),
        ("Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Performance", test_performance),
        ("CLI", test_cli),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Verification Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 Installation verification completed successfully!")
        print("   The package is ready for use.")
        return True
    else:
        print("\n⚠️  Some verification tests failed.")
        print("   Please check the error messages above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)