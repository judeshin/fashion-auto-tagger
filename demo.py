from auto_tagger import FashionAutoTagger

def main():
    """
    Demo script for Fashion Auto-Tagger.
    """
    # Fashion-specific keywords
    labels = [
        # Clothing types
        "hoodie", "pullover", "jacket", "shirt", "pants", "dress", "skirt", 
        "blazer", "coat", "sweater", "t-shirt", "jeans",
        # Details
        "zipper", "button", "pocket", "hood", "collar", "sleeve",
        # Materials/Patterns
        "denim", "leather", "cotton", "knit", "striped", "plaid", "floral",
        # Colors
        "black", "white", "blue", "red", "gray", "navy", "brown", "green"
    ]
    
    print("=" * 60)
    print("🏷️  Fashion Auto-Tagger Demo")
    print("=" * 60)
    print("\nThis tool generates relevant keywords for fashion images.")
    print("Powered by OpenAI's CLIP model.\n")
    
    # Initialize tagger
    tagger = FashionAutoTagger(labels=labels)
    
    # Get image path from user
    print("\n" + "-" * 60)
    image_path = input("Enter image path (or 'q' to quit): ").strip()
    
    if image_path.lower() == 'q':
        print("\nGoodbye!")
        return
    
    try:
        # Generate keywords
        print("\n⚙️  Processing image...")
        keywords = tagger.tag(
            image_path=image_path,
            top_k=5,
            threshold=0.185,
            show_confidence=True,
            debug=False
        )
        
        # Display results
        print("\n✨ Generated Keywords:")
        if keywords:
            for kw in keywords:
                print(f"   #{kw}")
        else:
            print("   No keywords found above threshold.")
            print("   Try lowering the threshold or using different labels.")
        
        print("\n" + "-" * 60)
        print("\n💡 Tip: Adjust threshold (0.15-0.25) for better results.")
        print("   Lower threshold = more keywords, but less accurate")
        print("   Higher threshold = fewer keywords, but more confident\n")
        
    except FileNotFoundError:
        print(f"\n❌ Error: Image file not found: {image_path}")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
