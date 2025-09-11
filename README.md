# NSRA PL14 Target Scoring Application

This project contains a Python application to automatically score NSRA PL14 target cards using computer vision. It includes a core scoring engine and a Qt5-based graphical user interface.

## Features

-   Scores .177 and .22 calibre shots.
-   Uses a 4-stage computer vision pipeline to localize the target, find the center, detect shots, and calculate scores.
-   Implements the official NSRA "inward gauging" rule.
-   Provides both a command-line interface and a graphical user interface.

## Setup

1.  **Clone the repository.**
2.  **Install dependencies:** It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Graphical User Interface (GUI)

This is the recommended way to use the application.

1.  **Run the GUI:**
    ```bash
    python main_gui.py
    ```
2.  Click the "Open Image" button to select a photo of your target.
3.  Select the correct calibre (.177 or .22).
4.  The annotated target and scores will be displayed.

### Command-Line Interface

The core engine can be run directly from the command line.

1.  **Run the script:**
    ```bash
    python scoring_engine/score.py --image /path/to/your/target.jpg --calibre 0.177
    ```
2.  The results will be printed to the console.
