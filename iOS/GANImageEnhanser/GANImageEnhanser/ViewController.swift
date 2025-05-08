//
//  ViewController.swift
//  GANImageEnhanser
//
//  Created by Jagjeetsingh Labana on 24/04/2025.
//

import UIKit
import Photos
import CoreML
import PhotosUI
import Vision

class ViewController: UIViewController {

    //MARK: OUTLETS
    @IBOutlet weak var buttonPickImage: UIButton!
    @IBOutlet weak var imageViewSelectedImage: UIImageView!
    @IBOutlet weak var imageViewEnhancedImage: UIImageView!
    @IBOutlet weak var viewProgress: UIView!
    @IBOutlet weak var viewSelectedImage: UIView!{
        didSet{
            viewSelectedImage.layer.borderColor = UIColor.black.cgColor
            viewSelectedImage.layer.borderWidth = 2
            viewSelectedImage.layer.cornerRadius = 8
        }
    }
    @IBOutlet weak var viewEnhancedImage: UIView!{
        didSet{
            viewEnhancedImage.layer.borderColor = UIColor.black.cgColor
            viewEnhancedImage.layer.borderWidth = 2
            viewEnhancedImage.layer.cornerRadius = 8
        }
    }
    
    private var model: MLModel?

    
    //MARK: CORE FUNCTIONS
    override func viewDidLoad() {
        super.viewDidLoad()
        viewSelectedImage.isHidden = true
        viewEnhancedImage.isHidden = true
        loadModel()
    }

    @IBAction func actionPickImage(_ sender: Any) {
        self.showImagePickerOptions()
    }
    
    //MARK: FUNCTIONS
    //Load Model To System
    private func loadModel() {
        guard let modelURL = Bundle.main.url(forResource: "ImageGenerator", withExtension: "mlmodelc") else {
            fatalError("❌ Could not find ImageGenerator.mlpackage")
        }
        do {
            let config = MLModelConfiguration()
            model = try MLModel(contentsOf: modelURL, configuration: config)
            print(" Model loaded.")
        } catch {
            fatalError("❌ Failed to load model: \(error)")
        }
    }
    
    //Image Picker based on sourced
    func imagePicker(sourceType: UIImagePickerController.SourceType) -> UIImagePickerController{
        let imagePicker = UIImagePickerController()
        imagePicker.sourceType = sourceType
        return imagePicker
    }
    
    //Show Image Picker option to user Either Photos or Camera
    func showImagePickerOptions(){
        let alert = UIAlertController(title: "Pick Image from", message: "", preferredStyle: .actionSheet)
        
        //Camera
        let camAction = UIAlertAction(title: "Camera", style: .default) { [weak self] (action) in
            guard let self = self else {
                return
            }
            let cameraImagePicker = self.imagePicker(sourceType: .camera)
            cameraImagePicker.delegate = self
            self.present(cameraImagePicker, animated: true)
        }
        
        //Library
        let libraryAction = UIAlertAction(title: "Photos", style: .default) { [weak self] (action) in
            guard let self = self else {
                return
            }
            let libraryImagePicker = self.imagePicker(sourceType: .photoLibrary)
            libraryImagePicker.delegate = self
            self.present(libraryImagePicker, animated: true)
        }
        let cancelAction = UIAlertAction(title: "Cancel", style: .destructive)
        alert.addAction(camAction)
        alert.addAction(libraryAction)
        alert.addAction(cancelAction)
        self.present(alert, animated: true)
    }
    
    //Show Progress View
    func showProgress(_ value:Bool){
        self.viewProgress.isHidden = !value
    }
    
//MARK: CORE ML FUNCTIONS
    
    private func runModel(on image: UIImage) {
        self.showProgress(true)

        DispatchQueue.global().async {
            guard let resized = image.resize(to: CGSize(width: 128, height: 128)),
                  let inputArray = resized.toMLMultiArray() else {
                DispatchQueue.main.async {
                    self.showProgress(false)
                    print("❌ Failed to preprocess image.")
                }
                return
            }

            guard let model = self.model else {
                DispatchQueue.main.async {
                    self.showProgress(false)
                    print("❌ Model not loaded.")
                }
                return
            }

            do {
                let input = try MLDictionaryFeatureProvider(dictionary: ["input": inputArray])
                let output = try model.prediction(from: input)

                guard let firstFeature = output.featureNames.first,
                      let resultArray = output.featureValue(for: firstFeature)?.multiArrayValue else {
                    DispatchQueue.main.async {
                        self.showProgress(false)
                        print("❌ Output feature not found.")
                    }
                    return
                }

                let enhancedImage = resultArray.toUIImage()

                DispatchQueue.main.async {
                    self.showProgress(false)
                    
                    self.imageViewEnhancedImage.image = enhancedImage
                }

            } catch {
                DispatchQueue.main.async {
                    self.showProgress(false)
                    print("❌ Prediction failed: \(error)")
                }
            }
        }
    }
}

    /*func processLowResImage(_ image: UIImage) {
        let url =  Bundle.main.url(forResource: "ImageGenerator", withExtension: "mlmodelc")
        guard let modelURL = Bundle.main.url(forResource: "ImageGenerator", withExtension: "mlmodelc") else {
            print("❌ Model not found.")
            self.viewProgress.isHidden = true
            return
        }

        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            let model = try MLModel(contentsOf: modelURL, configuration: config)


            let inputArray = try preprocessImage(image)

            let input = try MLDictionaryFeatureProvider(dictionary: ["input": inputArray])
            let output = try model.prediction(from: input)
            print(output)
            let featureNames = output.featureNames
            print("📦 Output features:", featureNames)

            if let firstFeatureName = featureNames.first,
               let result = output.featureValue(for: firstFeatureName)?.multiArrayValue {
                print("Result Value: ",result[0], result[1], result[2], result[1000])

                let enhanced = result.toUIImage()
                
                DispatchQueue.main.async {
                    self.viewProgress.isHidden = true
                    self.imageViewEnhancedImage.image = enhanced
                }
            } else {
                DispatchQueue.main.async {
                    self.viewProgress.isHidden = true
                }
                print("❌ No output result found.")
            }
            
            
//            if let result = output.featureValue(for: "output")?.multiArrayValue {
//                let enhanced = result.toUIImage()
//                DispatchQueue.main.async {
//                    self.viewProgress.isHidden = true
//                    self.imageViewEnhancedImage.image = enhanced
//                }
//            } else {
//                self.viewProgress.isHidden = true
//                print("❌ Output result not found.")
//            }

        } catch {
            self.viewProgress.isHidden = true
            print("❌ Error during prediction: \(error)")
        }
    }*/



//MARK: EXTENSION
extension ViewController : UIImagePickerControllerDelegate,UINavigationControllerDelegate {
    // Load Image which user selects from gallery or click from camera
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        DispatchQueue.main.async {
            self.showProgress(true)
        }
        if let image = info[.originalImage] as? UIImage {
            self.imageViewSelectedImage.image = image
            self.viewSelectedImage.isHidden = false
            self.viewEnhancedImage.isHidden = false
        }
        self.dismiss(animated: true) {
            
            
            if let image = info[.originalImage] as? UIImage {
                self.runModel(on: image)
            }
        }
    }
}
