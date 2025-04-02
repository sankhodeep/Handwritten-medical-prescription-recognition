import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Class for preprocessing prescription images to enhance readability
    before text extraction.
    """
    
    def __init__(self):
        """Initialize the image preprocessor."""
        logger.info("Initializing image preprocessor")
    
    def load_image(self, image_path):
        """
        Load an image from the given path.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: The loaded image
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image from {image_path}")
            logger.info(f"Successfully loaded image from {image_path}")
            return image
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            raise
    
    def enhance_contrast(self, image):
        """
        Enhance the contrast of the image.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Contrast-enhanced image
        """
        try:
            # Convert to grayscale if not already
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            logger.info("Successfully enhanced image contrast")
            return enhanced
        except Exception as e:
            logger.error(f"Error enhancing contrast: {str(e)}")
            raise
    
    def remove_noise(self, image):
        """
        Remove noise from the image.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Denoised image
        """
        try:
            # Apply bilateral filter to remove noise while preserving edges
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
            
            logger.info("Successfully removed noise from image")
            return denoised
        except Exception as e:
            logger.error(f"Error removing noise: {str(e)}")
            raise
    
    def correct_skew(self, image):
        """
        Correct skew/rotation in the image.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Deskewed image
        """
        try:
            # Convert to binary image
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find all contours
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest contour
            if not contours:
                logger.warning("No contours found for skew correction")
                return image
                
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Find the minimum area rectangle that encloses the contour
            rect = cv2.minAreaRect(largest_contour)
            angle = rect[2]
            
            # Adjust angle
            if angle < -45:
                angle = 90 + angle
            
            # Rotate the image
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            logger.info(f"Successfully corrected skew with angle {angle}")
            return rotated
        except Exception as e:
            logger.error(f"Error correcting skew: {str(e)}")
            # Return original image if skew correction fails
            return image
    
    def segment_handwritten_text(self, image):
        """
        Segment handwritten text from printed text.
        This is a simplified implementation that would need to be replaced
        with a proper segmentation model like Mask R-CNN or U-Net in production.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Image with segmented handwritten text
        """
        try:
            # This is a placeholder for a more sophisticated segmentation model
            # In a real implementation, this would use a trained model
            
            # For now, we'll use adaptive thresholding to separate text from background
            # This is not a true segmentation of handwritten vs printed text
            adaptive_thresh = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Dilate to connect nearby text
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(adaptive_thresh, kernel, iterations=1)
            
            logger.info("Applied simplified text segmentation")
            return dilated
        except Exception as e:
            logger.error(f"Error segmenting handwritten text: {str(e)}")
            raise
    
    def preprocess(self, image_path):
        """
        Preprocess the image to enhance readability.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Preprocessed image
            str: Path to the saved preprocessed image
        """
        try:
            # Load the image
            image = self.load_image(image_path)
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Enhance contrast
            enhanced = self.enhance_contrast(gray)
            
            # Remove noise
            denoised = self.remove_noise(enhanced)
            
            # Correct skew
            deskewed = self.correct_skew(denoised)
            
            # Segment handwritten text (simplified)
            segmented = self.segment_handwritten_text(deskewed)
            
            # Save the preprocessed image
            output_path = image_path.replace('.', '_preprocessed.')
            cv2.imwrite(output_path, segmented)
            
            logger.info(f"Successfully preprocessed image and saved to {output_path}")
            return segmented, output_path
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise