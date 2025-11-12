# Fashion Auto-Tagger: CLIP-based Automated Keyword Generation for Fashion Images

A proof-of-concept tool that automatically generates relevant keywords for fashion product images using OpenAI's CLIP model.

## 🎯 Background

As a Product Manager at CLO Virtual Fashion, I wanted to explore how AI could streamline the manual process of tagging 3D-rendered fashion items in our CLO-SET platform. This PoC demonstrates that automated keyword generation is technically feasible.

## 🚀 What It Does

**Input:** Fashion product image (PNG/JPG)  
**Output:** Relevant keywords with confidence scores

```
Input: hoodie.png
Output: ['hoodie(0.32)', 'pullover(0.30)', 'hood(0.29)', 'jacket(0.27)', 'shirt(0.27)']
```

## 🛠️ Tech Stack

- **CLIP (ViT-B/32)** - OpenAI's vision-language model
- **PyTorch** - Deep learning framework
- **Python 3.10+**

## 📦 Installation

```bash
# Create conda environment
conda create -n fashion_tag python=3.10
conda activate fashion_tag

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Basic Usage

```python
from auto_tagger import FashionAutoTagger

# Define candidate keywords
labels = [
    "hoodie", "jacket", "shirt", "pants",
    "zipper", "button", "pocket", "hood",
    "denim", "cotton", "black", "blue"
]

# Initialize tagger
tagger = FashionAutoTagger(labels=labels)

# Tag an image
keywords = tagger.tag(
    image_path="your_image.png",
    top_k=5,
    threshold=0.185,
    show_confidence=True
)

print(keywords)
# Output: ['hoodie(0.32)', 'pullover(0.30)', 'hood(0.29)', ...]
```

### Run Demo

```bash
python demo.py
```

## 🎓 Key Learnings

1. **CLIP requires predefined keyword lists** - It's a classification model, not a generative one
2. **Similarity scores typically range 0.15-0.35** - Requires careful threshold tuning
3. **Template averaging improves robustness** - Using multiple prompt templates per label
4. **Fast inference** - ~0.1 seconds per image on CPU

## ⚙️ Parameters

- `labels` (List[str]): Candidate keywords to search for
- `top_k` (int): Maximum number of keywords to return (default: 5)
- `threshold` (float): Minimum similarity score (default: 0.185, range: 0-1)
- `show_confidence` (bool): Include confidence scores in output (default: True)

## 🔮 Future Improvements

- [ ] Integrate with GPT-4V for free-form keyword generation
- [ ] Fine-tune on fashion-specific dataset
- [ ] Add support for multi-language keywords
- [ ] Build batch processing pipeline
- [ ] Deploy as API service

## 👤 About

Created by **Jude** - Product Manager at CLO Virtual Fashion

- Fashion design background (Bunka Fashion College, Tokyo)
- 5+ years in product management
- Passionate about AI × Fashion

## 📝 License

MIT License - feel free to use and modify!

---

**Note:** This is a proof-of-concept built in a few hours to explore technical feasibility. Production implementation would require significant additional work.
