import main

# Expose main.app for Hugging Face Spaces Gradio / FastAPI launcher
app = main.app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
