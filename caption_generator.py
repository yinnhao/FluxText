import argparse
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration


class CaptionGenerator:
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b"):
        """
        Initialize the caption generator.
        
        Args:
            model_name: HuggingFace model name for BLIP2
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize BLIP model for caption generation
        print(f"Loading BLIP2 model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.blipmodel = Blip2ForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16
        )
        self.blipmodel.to(self.device, torch.float16)
        print(f"Model loaded on device: {self.device}")
    
    def generate_caption(self, image_path: str, max_new_tokens: int = 20) -> str:
        """
        Generate a basic caption for the image.
        
        Args:
            image_path: Path to the input image
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Generated caption string
        """
        # Load and process image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)
        
        # Generate caption
        generated_ids = self.blipmodel.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        return generated_text
    
    def generate_caption_with_text(self, image_path: str, text: str, max_new_tokens: int = 20) -> str:
        """
        Generate caption for image with specified text content.
        
        Args:
            image_path: Path to the input image
            text: Text that should appear in the image
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Caption string formatted for text editing task
        """
        basic_caption = self.generate_caption(image_path, max_new_tokens)
        caption = f'{basic_caption}, that reads "{text}"'
        return caption
    
    def generate_caption_from_array(self, image_array: np.ndarray, text: str, max_new_tokens: int = 20) -> str:
        """
        Generate caption from numpy array with specified text.
        
        Args:
            image_array: Numpy array representing the image
            text: Text that should appear in the image
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Caption string formatted for text editing task
        """
        image_pil = Image.fromarray(image_array)
        inputs = self.processor(image_pil, return_tensors="pt").to(self.device, torch.float16)

        generated_ids = self.blipmodel.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        caption = f'{generated_text}, that reads "{text}"'
        return caption


def main():
    """Example usage of the CaptionGenerator."""
    parser = argparse.ArgumentParser(description="Generate captions for images with optional text content")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--text", type=str, default=None, help="Text content to include in caption")
    parser.add_argument("--output_file", type=str, default=None, help="File to save the caption (optional)")
    parser.add_argument("--model_name", type=str, default="Salesforce/blip2-opt-2.7b", 
                       help="HuggingFace model name for BLIP2")
    parser.add_argument("--max_tokens", type=int, default=20, help="Maximum number of new tokens to generate")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = CaptionGenerator(model_name=args.model_name)
    
    # Generate caption
    if args.text:
        caption = generator.generate_caption_with_text(
            image_path=args.image_path,
            text=args.text,
            max_new_tokens=args.max_tokens
        )
        print(f"Caption with text '{args.text}': {caption}")
    else:
        caption = generator.generate_caption(
            image_path=args.image_path,
            max_new_tokens=args.max_tokens
        )
        print(f"Basic caption: {caption}")
    
    # Save to file if specified
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(caption)
        print(f"Caption saved to: {args.output_file}")


if __name__ == "__main__":
    main() 