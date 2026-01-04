# Quick Start Guide

## Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Your First Example
```bash
python lime_example.py
```

### Step 3: View the Results
Check the generated `lime_explanation.png` file to see the visualization!

## What Each Script Does

| Script | Dataset | Model | Purpose |
|--------|---------|-------|---------|
| `lime_example.py` | Iris (flowers) | Random Forest | Basic LIME introduction |
| `lime_advanced.py` | Wine quality | Gradient Boosting | Multiple instance explanations |
| `lime_text_example.py` | Movie reviews | Naive Bayes | Text classification |

## Expected Output

When you run `lime_example.py`, you'll see:
- Dataset information (120 training, 30 test samples)
- Model accuracy (should be ~100%)
- Prediction details for a flower sample
- LIME explanation showing which features mattered most
- A saved visualization (PNG file)

## Troubleshooting

**Problem**: `ModuleNotFoundError`  
**Solution**: Run `pip install -r requirements.txt`

**Problem**: No visualization appears  
**Solution**: The visualization is saved as a file (e.g., `lime_explanation.png`)

**Problem**: Script runs but no output  
**Solution**: Check that matplotlib is installed correctly

## Next Steps

1. ✅ Run all three examples
2. ✅ Read the full [README.md](README.md)
3. ✅ Study the [DOCUMENTATION.md](DOCUMENTATION.md) for deeper understanding
4. ✅ Modify the code to experiment with different parameters
5. ✅ Try using your own dataset

## Learning Path

**Beginner** → Start with `lime_example.py`  
**Intermediate** → Try `lime_advanced.py`  
**Advanced** → Explore `lime_text_example.py` and modify parameters

---

**Need help?** Check the [DOCUMENTATION.md](DOCUMENTATION.md) file for detailed explanations!
