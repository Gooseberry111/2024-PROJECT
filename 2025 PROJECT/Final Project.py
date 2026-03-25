import numpy as np
import cv2
import pywt
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
from skimage.metrics import peak_signal_noise_ratio as psnr
import hashlib
import os
import csv
from PIL import Image, ImageEnhance

class MedicalImageWatermarker:
    def __init__(self):
        self.wavelet = 'haar'
        self.decomposition_level = 3
        self.output_base = "watermark_output"
        self.ensure_directories()
        
    def ensure_directories(self):
        
        try:
            os.makedirs(os.path.join(self.output_base, "watermark_keys"), exist_ok=True)
            os.makedirs(os.path.join(self.output_base, "extracted"), exist_ok=True)
            os.makedirs(os.path.join(self.output_base, "attacked"), exist_ok=True)
        except Exception as e:
            print(f"Error creating directories: {str(e)}")
            exit(1)
    
    def get_output_path(self, *args):
        return os.path.join(self.output_base, *args)
    
    def load_images(self, host_path, watermark_path):
        try:
            host_img = cv2.imread(host_path, cv2.IMREAD_GRAYSCALE)
            watermark_img = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
            
            if host_img is None or watermark_img is None:
                raise ValueError("Could not load one or both images")
                
            # Resize watermark to match host dimensions
            watermark_img = cv2.resize(watermark_img, (host_img.shape[1], host_img.shape[0]))
            return host_img, watermark_img
        except Exception as e:
            print(f"Error loading images: {str(e)}")
            exit(1)

    def apply_modwt(self, img):
        """3-level MODWT decomposition using Haar wavelet"""
        coeffs = pywt.wavedec2(img, self.wavelet, level=self.decomposition_level, mode='periodization')
        return coeffs[0]  # Return LL3 subband

    def apply_tangent_transform(self, LL_subband):
        return np.arctan(LL_subband)
    
    def extract_dft_features(self, transformed_img):
        dft_shift = np.fft.fftshift(np.fft.fft2(transformed_img))
        magnitude = np.log1p(np.abs(dft_shift))
        phase = np.angle(dft_shift)
        mag_thresh = np.percentile(magnitude, 75)
        hybrid_feature = ((magnitude > mag_thresh) & (phase > 0)).astype(np.uint8) * 255
        return hybrid_feature
    
    def generate_watermark_key(self, host_img, watermark_img, secret_text):
        # Feature extraction pipeline
        LL3 = self.apply_modwt(host_img)
        TT = self.apply_tangent_transform(LL3)
        binary_feature = self.extract_dft_features(TT)
        _, watermark_binary = cv2.threshold(watermark_img, 128, 255, cv2.THRESH_BINARY)
        secret_hash = hashlib.sha256(secret_text.encode()).hexdigest()
        secret_bin = ''.join(format(int(char, 16), '04b') for char in secret_hash)
        secret_array = np.array([int(bit) for bit in secret_bin], dtype=np.uint8) * 255
        h, w = binary_feature.shape
        watermark_binary = cv2.resize(watermark_binary, (w, h))
        secret_array = np.resize(secret_array, (h, w)).astype(np.uint8)
        combined = np.bitwise_xor(watermark_binary, secret_array)
        watermark_key = np.bitwise_xor(binary_feature, combined)
        
        return watermark_key
    
    def extract_watermark(self, test_img, watermark_key, secret_text):
        LL3_test = self.apply_modwt(test_img)
        TT_test = self.apply_tangent_transform(LL3_test)
        binary_feature_test = self.extract_dft_features(TT_test)
        watermark_key = cv2.resize(watermark_key, 
                                 (binary_feature_test.shape[1], binary_feature_test.shape[0]))
        combined = np.bitwise_xor(binary_feature_test, watermark_key)
        
        secret_hash = hashlib.sha256(secret_text.encode()).hexdigest()
        secret_bin = ''.join(format(int(char, 16), '04b') for char in secret_hash)
        secret_array = np.array([int(bit) for bit in secret_bin], dtype=np.uint8) * 255
        secret_array = np.resize(secret_array, combined.shape).astype(np.uint8)
        
        extracted_watermark = np.bitwise_xor(combined, secret_array)
        _, extracted_binary = cv2.threshold(extracted_watermark, 128, 255, cv2.THRESH_BINARY)
        
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(extracted_binary, -1, kernel)
    
    def verify_authenticity(self, test_img, watermark_key, original_watermark, secret_text, test_id="test"):
        try:
            extracted = self.extract_watermark(test_img, watermark_key, secret_text)
            original_resized = cv2.resize(original_watermark, (extracted.shape[1], extracted.shape[0]))
            _, original_binary = cv2.threshold(original_resized, 128, 255, cv2.THRESH_BINARY)
            
            metrics = {
                'ssim': ssim(extracted, original_binary, data_range=255),
                'correlation': self.calculate_correlation(extracted, original_binary),
                'psnr': psnr(original_binary, extracted, data_range=255),
                'ber': self.calculate_ber(original_binary, extracted),
                'ncc': self.calculate_ncc(original_binary, extracted),
                'mse': self.calculate_mse(original_binary, extracted),
                'mae': self.calculate_mae(original_binary, extracted),
                'gei': self.calculate_gei(extracted)
            }
            
            metrics['authentic'] = metrics['ssim'] > 0.6 and metrics['correlation'] > 0.5
            
            self.save_results(test_id, metrics, extracted)
            self.display_results(original_binary, extracted, metrics, test_id)
            
            return metrics
        except Exception as e:
            print(f"Error during verification: {str(e)}")
            return None
    
    def calculate_correlation(self, img1, img2):
        img1_flat = img1.flatten().astype(np.float64)
        img2_flat = img2.flatten().astype(np.float64)
        corr, _ = pearsonr(img1_flat, img2_flat)
        return corr
    
    def calculate_ber(self, original, extracted):
        original_flat = original.flatten() // 255
        extracted_flat = extracted.flatten() // 255
        return np.sum(original_flat != extracted_flat) / len(original_flat)
    
    def calculate_ncc(self, img1, img2):
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        numerator = np.sum(img1 * img2)
        denominator = np.sqrt(np.sum(img1**2) * np.sum(img2**2))
        return numerator / denominator if denominator != 0 else 0
    
    def calculate_mse(self, img1, img2):
        return np.mean((img1.astype(np.float64)) - (img2.astype(np.float64))) ** 2
    
    def calculate_mae(self, img1, img2):
        return np.mean(np.abs(img1.astype(np.float64) - img2.astype(np.float64)))
    
    def calculate_gei(self, img):
        return np.sum(img.astype(np.float64)) ** 2 / (img.shape[0] * img.shape[1])
    
    def save_results(self, test_id, metrics, extracted_watermark):
        try:
            csv_path = self.get_output_path("results.csv")
            
            if not os.path.exists(csv_path):
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Test ID', 'SSIM', 'Correlation', 'PSNR', 'BER', 
                                   'NCC', 'MSE', 'MAE', 'GEI', 'Authentic'])
            
            row = [test_id, metrics['ssim'], metrics['correlation'], 
                   metrics['psnr'], metrics['ber'], metrics['ncc'],
                   metrics['mse'], metrics['mae'], metrics['gei'], metrics['authentic']]
            
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            output_path = self.get_output_path("extracted", f"extracted_{test_id}.png")
            cv2.imwrite(output_path, extracted_watermark)
            
        except Exception as e:
            print(f"Error saving results: {str(e)}")

    def display_results(self, original, extracted, metrics, test_id):
        """Display comparison of watermarks"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(original, cmap='gray')
        plt.title(f"Original Watermark\nTest ID: {test_id}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(extracted, cmap='gray')
        plt.title(f"Extracted Watermark\nSSIM: {metrics['ssim']:.3f} | Corr: {metrics['correlation']:.3f}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nAuthentication Results for {test_id}:")
        print(f"SSIM: {metrics['ssim']:.4f}")
        print(f"Pearson Correlation: {metrics['correlation']:.4f}")
        print(f"PSNR: {metrics['psnr']:.2f} dB")
        print(f"Bit Error Rate: {metrics['ber']:.4f}")
        print(f"NCC: {metrics['ncc']:.4f}")
        print(f"MSE: {metrics['mse']:.2f}")
        print(f"MAE: {metrics['mae']:.2f}")
        print(f"GEI: {metrics['gei']:.2f}")
        print(f"Authentication Status: {'Authentic' if metrics['authentic'] else 'Tampered'}")

class RobustnessTester:
    def __init__(self):
        self.watermarker = MedicalImageWatermarker()
    
    def apply_jpeg_compression(self, img, quality=90):
        temp_path = os.path.join("/tmp" if os.name == 'posix' else os.getenv("TEMP"), "temp_jpeg.jpg")
        cv2.imwrite(temp_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        compressed = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
        os.remove(temp_path)
        return compressed
    
    def apply_salt_pepper_noise(self, img, amount=0.001):
        noisy = np.copy(img)
        # Salt mode
        num_salt = np.ceil(amount * img.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
        noisy[coords[0], coords[1]] = 255
        
        # Pepper mode
        num_pepper = np.ceil(amount * img.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
        noisy[coords[0], coords[1]] = 0
        
        return noisy
   
    '''def apply_cropping(self, img, crop_percent=0.02):
        """Apply cropping attack"""
        h, w = img.shape
        crop_h = int(h * crop_percent)
        crop_w = int(w * crop_percent)
        
        # Crop from all sides
        cropped = img[crop_h:h-crop_h, crop_w:w-crop_w]
        
        # Resize back to original dimensions
        return cv2.resize(cropped, (w, h))'''
    
    def apply_gaussian_blur(self, img, kernel_size=(3,3)):
        return cv2.GaussianBlur(img, kernel_size, 0)
    
    def test_robustness(self, host_img, watermark_key, original_watermark, secret_text):
        attacks = [
            ('no_attack', 'No Attack', {}),
            ('jpeg', 'JPEG Compression (Q90)', {'quality': 90}),
            ('salt_pepper', 'Salt & Pepper Noise (0.1%)', {'amount': 0.001}),
            ('gaussian_blur', 'Gaussian Blur (3x3)', {'kernel_size': (3,3)}),
            '''('cropping', 'Cropping (2%)', {'crop_percent': 0.05})'''
        ]
        
        print("\nStarting robustness tests...")
        results = []
        
        for attack_id, attack_name, params in attacks:
            try:
                print(f"\nRunning test: {attack_name}...")
                
                # Apply attack
                attacked_img = host_img.copy()
                if attack_id == 'jpeg':
                    attacked_img = self.apply_jpeg_compression(attacked_img, params['quality'])
                elif attack_id == 'salt_pepper':
                    attacked_img = self.apply_salt_pepper_noise(attacked_img, params['amount'])
                elif attack_id == 'gaussian_blur':
                    attacked_img = self.apply_gaussian_blur(attacked_img, params['kernel_size'])
                '''elif attack_id == 'cropping':
                    attacked_img = self.apply_cropping(attacked_img, params['crop_percent'])'''
                
                # Save attacked image
                attack_path = self.watermarker.get_output_path("attacked", f"{attack_id}.png")
                cv2.imwrite(attack_path, attacked_img)
                
                # Verify authenticity
                metrics = self.watermarker.verify_authenticity(
                    attacked_img, watermark_key, original_watermark, secret_text, attack_id)
                
                if metrics:
                    results.append((attack_name, metrics))
                    print(f"Completed: {attack_name}")
                    
            except Exception as e:
                print(f"Error during {attack_name} test: {str(e)}")
                continue
        
        self.print_summary(results)

    def print_summary(self, results):
        if not results:
            print("\nNo tests completed successfully")
            return
            
        print("\n" + "="*100)
        print("ROBUSTNESS TEST SUMMARY")
        print("="*100)
        print("{:<25} {:<8} {:<12} {:<10} {:<8} {:<10} {:<10} {:<10} {:<10}".format(
            "ATTACK", "SSIM", "CORR", "PSNR", "BER", "NCC", "MSE", "MAE", "STATUS"))
        print("-"*100)
        
        for name, metrics in results:
            status = "PASS" if metrics['authentic'] else "FAIL"
            print("{:<25} {:<8.3f} {:<12.3f} {:<10.2f} {:<8.4f} {:<10.4f} {:<10.2f} {:<10.2f} {:<10}".format(
                name, metrics['ssim'], metrics['correlation'], 
                metrics['psnr'], metrics['ber'], metrics['ncc'],
                metrics['mse'], metrics['mae'], status))
        
        print("="*100)
        
        passed = sum(1 for _, m in results if m['authentic'])
        total = len(results)
        robustness_score = (passed / total) * 100 if total > 0 else 0
        
        print(f"\nOVERALL ROBUSTNESS: {robustness_score:.1f}% ({passed}/{total} tests passed)")
        print("="*100)

if __name__ == "__main__":
    try:
        print("\nMODWT-BASED ZERO WATERMARKING FOR MEDICAL IMAGES")
        print("="*60)
       
        watermarker = MedicalImageWatermarker()
        tester = RobustnessTester()
        
        print("\nPlease provide the following inputs:")
        host_path = input("- Path to host medical image: ").strip()
        watermark_path = input("- Path to watermark image: ").strip()
        secret_text = input("- Secret authentication text: ").strip()
        
        print("\nLoading images...")
        host_img, watermark_img = watermarker.load_images(host_path, watermark_path)
        
        print("\nOptions:")
        print("1. Embed & Generate Watermark Key")
        print("2. Extract & Verify Authenticity")
        print("3. Test Robustness Against Attacks")
        choice = input("Select operation (1/2/3): ").strip()
        
        if choice == '1':
            print("\nGenerating watermark key...")
            watermark_key = watermarker.generate_watermark_key(host_img, watermark_img, secret_text)
            key_path = watermarker.get_output_path("watermark_keys", "watermark_key.npy")
            np.save(key_path, watermark_key)
            print(f"\nSUCCESS: Watermark key saved to {key_path}")
            
        elif choice in ['2', '3']:
            key_path = input("\nPath to watermark key (.npy file): ").strip()
            if not os.path.exists(key_path):
                print(f"ERROR: Watermark key not found at {key_path}")
                exit(1)
                
            watermark_key = np.load(key_path)
            
            if choice == '2':
                test_path = input("\nPath to test image: ").strip()
                test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
                if test_img is None:
                    print(f"ERROR: Could not decode test image at {test_path}")
                    exit(1)
                
                test_id = os.path.splitext(os.path.basename(test_path))[0]
                watermarker.verify_authenticity(test_img, watermark_key, watermark_img, secret_text, test_id)
                
            elif choice == '3':
                print("\nPreparing to test robustness...")
                tester.test_robustness(host_img, watermark_key, watermark_img, secret_text)
                
        else:
            print("Invalid choice. Please select 1, 2, or 3.")
            
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
    finally:
        print("\nOperation completed. Output files saved in:")
        print(f"- {watermarker.get_output_path()}")