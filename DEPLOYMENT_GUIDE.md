# 🚀 Vercel Deployment Guide

## 📋 Prerequisites

- GitHub account
- Vercel account
- Git installed
- Python 3.9+

## 📁 Final Project Structure

```
new AI 2 app/
├── api/
│   ├── app.py                 # Main Flask application
│   ├── vectorizer.pkl          # TF-IDF vectorizer
│   ├── category_model.pkl      # Category classifier
│   ├── priority_model.pkl      # Priority classifier
│   ├── category_encoder.pkl    # Category label encoder
│   └── priority_encoder.pkl    # Priority label encoder
├── templates/
│   └── index.html            # Frontend HTML
├── static/                   # Static assets (if needed)
├── requirements.txt           # Python dependencies
├── vercel.json             # Vercel configuration
└── prepare_models.py        # Model preparation script
```

## 🛠️ Step 1: Prepare Model Files

If you haven't already prepared the model files:

```bash
python prepare_models.py
```

This will extract and save all model components to the `api/` directory.

## 🛠️ Step 2: Initialize Git Repository

```bash
# Initialize git
git init

# Add all files
git add .

# Initial commit
git commit -m "Initial commit: AI Complaint Analyzer"
```

## 🛠️ Step 3: Create GitHub Repository

1. Go to [GitHub](https://github.com) and create a new repository
2. Name it: `ai-complaint-analyzer`
3. Copy the repository URL

## 🛠️ Step 4: Push to GitHub

```bash
# Add remote origin (replace with your repo URL)
git remote add origin https://github.com/yourusername/ai-complaint-analyzer.git

# Push to GitHub
git push -u origin main
```

## 🛠️ Step 5: Deploy to Vercel

### Option A: Using Vercel CLI

1. Install Vercel CLI:
```bash
npm i -g vercel
```

2. Login to Vercel:
```bash
vercel login
```

3. Deploy:
```bash
vercel --prod
```

### Option B: Using Vercel Dashboard

1. Go to [vercel.com](https://vercel.com)
2. Click "Add New Project"
3. Import your GitHub repository
4. Vercel will automatically detect the Python project
5. Click "Deploy"

## 🛠️ Step 6: Configure Environment Variables (if needed)

In Vercel dashboard, add any required environment variables:
- `PYTHON_VERSION`: `3.9`

## ✅ Verification

After deployment, your app will be available at:
`https://your-project-name.vercel.app`

### Test the deployment:

1. **Web Interface**: Visit the URL and test the form
2. **API Endpoint**: Test with curl:
```bash
curl -X POST https://your-project-name.vercel.app/predict \
  -H "Content-Type: application/json" \
  -d '{"complaint_text": "I was charged $99.99 for a service I didn'\''t subscribe to"}'
```

## 🔧 Troubleshooting

### Common Issues:

1. **Model Loading Errors**
   - Ensure all model files are in `api/` directory
   - Check file sizes (should not be 0 bytes)

2. **Import Errors**
   - Verify `requirements.txt` has all dependencies
   - Check Python version compatibility

3. **Build Failures**
   - Check Vercel build logs
   - Ensure `vercel.json` is correctly configured

4. **Runtime Errors**
   - Check Vercel function logs
   - Verify file paths are relative

### Debug Commands:

```bash
# Check model files
ls -la api/

# Test locally
cd api && python app.py
```

## 📊 Expected Performance

- **Cold Start**: ~2-3 seconds (first request)
- **Warm Requests**: <500ms
- **Memory Usage**: ~512MB
- **Accuracy**: Same as local training results

## 🔄 Updates and Maintenance

### To Update Models:

1. Retrain models locally
2. Run `python prepare_models.py`
3. Commit and push changes
4. Vercel will auto-redeploy

### To Monitor:

- Check Vercel Analytics
- Monitor function logs
- Set up error alerts

## 🎯 Production Tips

1. **Add Domain**: Configure custom domain in Vercel dashboard
2. **SSL Certificate**: Auto-provided by Vercel
3. **CDN**: Automatic global distribution
4. **Scaling**: Automatic scaling with Vercel

## 📞 Support

- Vercel Documentation: https://vercel.com/docs
- GitHub Issues: Report deployment issues
- Performance: Monitor Vercel analytics

---

🎉 **Your AI Complaint Analyzer is now production-ready on Vercel!**
