//
//  extension.swift
//  GANImageEnhanser
//
//  Created by Jagjeetsingh Labana on 25/04/2025.
//

import UIKit
import Accelerate
import CoreML
import Vision
// MARK: - UIImage Helpers
extension UIImage {
    //Resize image to Specified Size
    func resize(to targetSize: CGSize) -> UIImage? {
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
        return renderer.image { _ in
            self.draw(in: CGRect(origin: .zero, size: targetSize))
        }
    }

    // Comvert Image to multiArray for Preprocess on image
    func toMLMultiArray() -> MLMultiArray? {
        guard let resized = self.cgImage else { return nil }

        do {
            let array = try MLMultiArray(shape: [1, 3, NSNumber(value: resized.height), NSNumber(value: resized.width)], dataType: .float32)

            guard let context = CGContext(
                data: nil,
                width: resized.width,
                height: resized.height,
                bitsPerComponent: 8,
                bytesPerRow: resized.width * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
            ) else {
                return nil
            }

            context.draw(resized, in: CGRect(x: 0, y: 0, width: resized.width, height: resized.height))

            guard let buffer = context.data else { return nil }
            let ptr = buffer.bindMemory(to: UInt8.self, capacity: resized.width * resized.height * 4)

            for y in 0..<resized.height {
                for x in 0..<resized.width {
                    let idx = (y * resized.width + x) * 4

                    let r = Float(ptr[idx + 1]) / 255.0
                    let g = Float(ptr[idx + 2]) / 255.0
                    let b = Float(ptr[idx + 3]) / 255.0

                    array[[0, 0, NSNumber(value: y), NSNumber(value: x)]] = NSNumber(value: r)
                    array[[0, 1, NSNumber(value: y), NSNumber(value: x)]] = NSNumber(value: g)
                    array[[0, 2, NSNumber(value: y), NSNumber(value: x)]] = NSNumber(value: b)
                }
            }

            return array
        } catch {
            print("❌ MLMultiArray creation failed: \(error)")
            return nil
        }
    }
}

// MARK:  MLMultiArray Helpers
extension MLMultiArray {
    //Convert back MultiArray to Image
    func toUIImage() -> UIImage? {
        let channels = self.shape[self.shape.count - 3].intValue
        let height = self.shape[self.shape.count - 2].intValue
        let width = self.shape[self.shape.count - 1].intValue

        guard channels == 3 else {
            print("❌ Only RGB output supported.")
            return nil
        }

        let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(self.dataPointer))
        let buffer = UnsafeBufferPointer(start: ptr, count: width * height * channels)

        // Step 1: Auto-detect min and max values in the model output
        var minValue: Float = Float.greatestFiniteMagnitude
        var maxValue: Float = -Float.greatestFiniteMagnitude

        for v in buffer {
            minValue = Swift.min(minValue, v)
            maxValue = Swift.max(maxValue, v)
        }

        print("Model output range detected: [\(minValue), \(maxValue)]")

        // Step 2: Decide scaling strategy
        let needsNormalization = !(minValue >= 0 && maxValue <= 1)

        // Step 3: Create pixel buffer
        var pixelData = [UInt8](repeating: 0, count: width * height * 4)

        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x

                var r = buffer[idx]
                var g = buffer[width * height + idx]
                var b = buffer[2 * width * height + idx]

                //If model outputs [-1,1], map to [0,1]
                if needsNormalization {
                    r = (r * 0.5) + 0.5
                    g = (g * 0.5) + 0.5
                    b = (b * 0.5) + 0.5
                }

                // Clip to [0,1] to avoid overflow
                r = min(max(r, 0), 1)
                g = min(max(g, 0), 1)
                b = min(max(b, 0), 1)

                pixelData[idx * 4 + 0] = UInt8(r * 255)
                pixelData[idx * 4 + 1] = UInt8(g * 255)
                pixelData[idx * 4 + 2] = UInt8(b * 255)
                pixelData[idx * 4 + 3] = 255  // Full alpha
            }
        }

        // Step 4: Create CGImage
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
            .union(.byteOrder32Big)

        let data = Data(pixelData)
        guard let providerRef = CGDataProvider(data: data as CFData) else { return nil }
        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: providerRef,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        ) else {
            return nil
        }

        return UIImage(cgImage: cgImage)
    }
}


