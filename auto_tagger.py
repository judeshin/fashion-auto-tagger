import torch
import clip
from PIL import Image
from typing import List

class FashionAutoTagger:
    """
    CLIP-based fashion image auto-tagger.
    
    This class uses OpenAI's CLIP model to automatically generate relevant
    keywords for fashion product images based on predefined candidate labels.
    """
    
    def __init__(self, labels: List[str]):
        """
        Initialize the tagger with candidate labels.
        
        Args:
            labels: List of candidate keywords to search for
        """
        self.labels = labels
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.text_embeddings = self._encode_labels()
        print(f"Tagger initialized (device: {self.device})")
    
    def _encode_labels(self) -> torch.Tensor:
        """
        Encode labels into CLIP embeddings using multiple templates.
        
        Returns:
            Normalized text embeddings tensor
        """
        templates = [
            "a photo of a {}",
            "a close-up of a {}",
            "a product photo of {}"
        ]
        
        with torch.no_grad():
            embeddings = []
            for label in self.labels:
                texts = [t.format(label) for t in templates]
                tokens = clip.tokenize(texts).to(self.device)
                text_features = self.model.encode_text(tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                embeddings.append(text_features.mean(dim=0))
            
            result = torch.stack(embeddings)
            return result / result.norm(dim=-1, keepdim=True)
    
    def tag(
        self, 
        image_path: str, 
        top_k: int = 5, 
        threshold: float = 0.185, 
        show_confidence: bool = True,
        debug: bool = False
    ) -> List[str]:
        """
        Generate keywords for an image.
        
        Args:
            image_path: Path to the image file
            top_k: Maximum number of keywords to return
            threshold: Minimum similarity score (0-1)
            show_confidence: Include confidence scores in output
            debug: Print debugging information
        
        Returns:
            List of keywords (with optional confidence scores)
        """
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ self.text_embeddings.T).squeeze(0)
            
            if debug:
                print(f"\nDebug Info:")
                print(f"Similarity range: min={similarity.min():.3f}, max={similarity.max():.3f}")
                top_indices = similarity.topk(min(10, len(self.labels))).indices
                print(f"Top 10 matches:")
                for idx in top_indices:
                    print(f"  {self.labels[idx]}: {similarity[idx]:.3f}")
        
        results = []
        for i in range(len(self.labels)):
            score = float(similarity[i])
            if score > threshold:
                results.append((self.labels[i], score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]
        
        if show_confidence:
            return [f"{tag}({conf:.2f})" for tag, conf in results]
        else:
            return [tag for tag, _ in results]
