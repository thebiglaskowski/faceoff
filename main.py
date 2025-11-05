"""
FaceOff - AI Face Swapper
Main entry point for the application.
"""
from ui.app import create_app

if __name__ == "__main__":
    demo = create_app()
    demo.launch(share=True)
