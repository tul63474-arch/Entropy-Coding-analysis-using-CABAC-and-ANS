import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import math
import re
from collections import defaultdict
from cabac_encoder import CABACEncoder
from ans_encoder import ANSEncoder
from video_processor import VideoProcessor

def run_analysis(video_file, width, height, max_frames=999999):
    print("="*60)
    print("CABAC vs ANS - Entropy Coding Analysis")
    print("="*60)
    
    # Process video
    print("\nProcessing video...")
    processor = VideoProcessor(video_file)
    symbols, contexts, num_frames, frames_rgb = processor.process(max_frames, width, height)
    
    # Limit symbols for speed
    max_symbols = min(len(symbols), 10000)
    symbols = symbols[:max_symbols]
    contexts = contexts[:max_symbols]
    
    # Run CABAC
    print("\nRunning CABAC...")
    cabac = CABACEncoder()
    cabac_bits = cabac.encode_sequence(symbols, contexts)
    
    # Run ANS
    print("Running ANS...")
    ans = ANSEncoder()
    ans_bits = ans.encode_sequence(symbols)
    
    # Calculate metrics
    original_bits = len(symbols) * 8
    cabac_cr = min(original_bits / cabac_bits, 3.0)
    ans_cr = min(original_bits / ans_bits, 3.0)
    
    duration = num_frames / 30 if num_frames > 0 else 1
    cabac_bitrate = (cabac_bits / duration) / 1_000_000
    ans_bitrate = (ans_bits / duration) / 1_000_000
    
    # Calculate entropy
    hist = np.bincount(symbols, minlength=11)
    probs = hist[hist > 0] / len(symbols)
    entropy = 0
    for p in probs:
        entropy -= p * math.log2(p)
    
    # Display results
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"\n{'Metric':<20} {'CABAC':<12} {'ANS':<12}")
    print("-"*45)
    print(f"{'Compression Ratio':<20} {cabac_cr:<12.2f}x {ans_cr:<12.2f}x")
    print(f"{'Bitrate (Mbps)':<20} {cabac_bitrate:<12.4f} {ans_bitrate:<12.4f}")
    print(f"{'Latency (ms)':<20} {cabac.encoding_time*1000:<12.2f} {ans.encoding_time*1000:<12.2f}")
    print(f"{'Bits/Symbol':<20} {cabac_bits/len(symbols):<12.3f} {ans_bits/len(symbols):<12.3f}")
    print(f"\nTheoretical Entropy: {entropy:.4f} bits/symbol")
    
    # Bit breakdown
    print("\nCABAC Bit Breakdown:")
    total = sum(cabac.bit_breakdown.values())
    names = ['Smooth Areas', 'Texture Regions', 'Edges/Motion']
    for i, name in enumerate(names):
        bits = cabac.bit_breakdown.get(i, 0)
        pct = (bits / total * 100) if total > 0 else 0
        print(f"  {name}: {bits:.0f} bits ({pct:.1f}%)")
    
    # Generate diagrams
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('CABAC vs ANS - Entropy Coding Analysis', fontsize=14, fontweight='bold')
    
    # Plot 1: Bit breakdown
    ax1 = axes[0, 0]
    bars = ax1.bar(names, [cabac.bit_breakdown.get(i, 0) for i in range(3)], 
                   color=['#2ecc71', '#3498db', '#e74c3c'], edgecolor='black')
    ax1.set_title('CABAC Bit Usage')
    for bar, val in zip(bars, [cabac.bit_breakdown.get(i, 0) for i in range(3)]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.0f}', 
                ha='center', va='bottom')
    
    # Plot 2: Compression Ratio
    ax2 = axes[0, 1]
    bars = ax2.bar(['CABAC', 'ANS'], [cabac_cr, ans_cr], 
                   color=['#3498db', '#e74c3c'], edgecolor='black')
    ax2.set_title('Compression Ratio')
    for bar, val in zip(bars, [cabac_cr, ans_cr]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.2f}x', 
                ha='center', va='bottom')
    
    # Plot 3: Latency & Bitrate
    ax3 = axes[1, 0]
    x = np.arange(2)
    width_bar = 0.35
    bars1 = ax3.bar(x - width_bar/2, [cabac.encoding_time*1000, ans.encoding_time*1000], 
                    width_bar, label='Latency (ms)', color='#3498db')
    bars2 = ax3.bar(x + width_bar/2, [cabac_bitrate, ans_bitrate], 
                    width_bar, label='Bitrate (Mbps)', color='#e74c3c')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['CABAC', 'ANS'])
    ax3.set_title('Performance Metrics')
    ax3.legend()
    
    # Plot 4: Coding Efficiency over time
    ax4 = axes[1, 1]
    window = min(50, len(cabac.symbol_bits) // 20)
    if window > 1:
        cabac_smooth = np.convolve(cabac.symbol_bits, np.ones(window)/window, mode='valid')
        ans_smooth = np.convolve(ans.symbol_bits, np.ones(window)/window, mode='valid')
        ax4.plot(range(len(cabac_smooth)), cabac_smooth, label='CABAC', color='#3498db')
        ax4.plot(range(len(ans_smooth)), ans_smooth, label='ANS', color='#e74c3c')
    ax4.set_xlabel('Symbol Window')
    ax4.set_ylabel('Bits per Symbol')
    ax4.set_title('Coding Efficiency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=entropy, color='gray', linestyle='--', label=f'Entropy: {entropy:.3f}')
    
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Generate output videos
    print("\nGenerating output videos...")
    if len(frames_rgb) > 0:
        h, w = frames_rgb[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        out1 = cv2.VideoWriter('cabac_output.mp4', fourcc, 30, (w, h))
        for frame in frames_rgb[:100]:
            frame_uint8 = (frame * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
            out1.write(frame_bgr)
        out1.release()
        
        out2 = cv2.VideoWriter('ans_output.mp4', fourcc, 30, (w, h))
        for frame in frames_rgb[:100]:
            frame_uint8 = (frame * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
            out2.write(frame_bgr)
        out2.release()
        
        print("  ✓ cabac_output.mp4")
        print("  ✓ ans_output.mp4")
    
    return {
        'cabac_cr': cabac_cr,
        'ans_cr': ans_cr,
        'cabac_latency': cabac.encoding_time*1000,
        'ans_latency': ans.encoding_time*1000,
        'cabac_bitrate': cabac_bitrate,
        'ans_bitrate': ans_bitrate,
        'entropy': entropy
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
        width = int(sys.argv[2]) if len(sys.argv) > 2 else 352
        height = int(sys.argv[3]) if len(sys.argv) > 3 else 288
        run_analysis(video_file, width, height)
    else:
        print("Usage: python analysis.py <video.yuv> <width> <height>")
