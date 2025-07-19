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
    
    def _preprocess_image(self, image):
        """
        Preprocess image to ensure compatibility with BLIP2.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # For BLIP2 stability, ensure image dimensions are divisible by 32
        # and use a standard size that works well with the model
        target_size = 384  # Common size that works well with BLIP2
        image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        return image
    
    def generate_caption(self, image_path: str, max_new_tokens: int = 20) -> str:
        """
        Generate a basic caption for the image.
        
        Args:
            image_path: Path to the input image
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Generated caption string
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            image = self._preprocess_image(image)
            
            # Process image with specific parameters for stability
            inputs = self.processor(
                image, 
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device, torch.float16)
            
            # Generate caption with conservative parameters
            with torch.no_grad():
                generated_ids = self.blipmodel.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    min_length=1,
                    do_sample=False,
                    num_beams=1,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            # Clean up the generated text
            if generated_text.startswith("Question: What is in this image? Answer:"):
                generated_text = generated_text.replace("Question: What is in this image? Answer:", "").strip()
            
            # Additional cleanup
            if not generated_text or generated_text.lower() in ['', 'a', 'an', 'the']:
                generated_text = "an image"
            
            return generated_text
            
        except Exception as e:
            print(f"Error generating caption: {e}")
            # Return a fallback caption
            return "an image"
    
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
        try:
            image_pil = Image.fromarray(image_array)
            image_pil = self._preprocess_image(image_pil)
            
            # Process image with specific parameters for stability
            inputs = self.processor(
                image_pil, 
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device, torch.float16)

            with torch.no_grad():
                generated_ids = self.blipmodel.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    min_length=1,
                    do_sample=False,
                    num_beams=1,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            # Clean up the generated text
            if generated_text.startswith("Question: What is in this image? Answer:"):
                generated_text = generated_text.replace("Question: What is in this image? Answer:", "").strip()
            
            # Additional cleanup
            if not generated_text or generated_text.lower() in ['', 'a', 'an', 'the']:
                generated_text = "an image"

            caption = f'{generated_text}, that reads "{text}"'
            return caption
            
        except Exception as e:
            print(f"Error generating caption from array: {e}")
            # Return a fallback caption
            return f'an image, that reads "{text}"'


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