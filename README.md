# SPSStoExcel
This is a super simple "app" that let's you view SPSS Files and export the data with or without the labels to Excel. 

Here's the link to the online version:
https://spsstoexcel.streamlit.app/
Unfortunatlely it get's shut down now and then, due to data usage limits of the provider (Streamlit, awesome open source python framework).

I think that SPSS still has a very efficent way for viewing and analyzing data sets. The possibility to easily switch betwen raw data, labelled data and variable definitions is still unique and very convenient. In this app you can at least export the different SPSS views to excel. i also added some basic options for statistical analysis and tabulations, hope to improve the possibilities over time.




<a href="https://www.buymeacoffee.com/thomafors8" target="_blank">
  <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" >
</a>


## Local Usage

To use this app locally on your computer, follow these steps:

Running a Streamlit app that you found on GitHub is a great way to try out data tools, dashboards, and web applications. If you have never done this before, the process might seem a bit intimidating, but it breaks down into a few straightforward steps.

Here is a step-by-step, beginner-friendly guide to getting that app running locally on your computer.

---

## Prerequisites: Quick Check

At first, make sure you have **Python** installed on your computer.

* You can check by opening your terminal (Mac/Linux) or Command Prompt/PowerShell (Windows) and typing: `python --version` (or `python3 --version`).
* If you don't have it, download and install it from [python.org](https://www.python.org/).

---

## Step 1: Download the Code from GitHub

There are two ways to get the code onto your machine: the easy way (ZIP download) or the developer way (Git).

### Option A: Download as a ZIP file (Easiest)

1. Go to the GitHub repository page for the Streamlit app.
2. Click the bright green **`<> Code`** button near the top right of the file list.
3. Click **Download ZIP**.
4. Once downloaded, extract (unzip) the file somewhere easy to find, like your Desktop or Documents folder.

### Option B: Clone using Git (If you have Git installed)

If you prefer using the command line, open your terminal, navigate to where you want the project folder to live, and run:

```bash
git clone https://github.com/username/repository-name.git

```

*(Replace the URL with the actual URL of the GitHub page).*

---

## Step 2: Open Your Terminal and Navigate to the Folder

Streamlit apps are launched using the command line.

1. Open your **Terminal** (Mac) or **Command Prompt/PowerShell** (Windows).
2. Use the `cd` (change directory) command to navigate into the folder you just extracted or cloned.

> **Windows Example:**
> ```cmd
> cd C:\Users\YourName\Desktop\my-streamlit-app
> 
> ```
> 
> 

> **Mac/Linux Example:**
> ```bash
> cd ~/Desktop/my-streamlit-app
> 
> ```
> 
> 

*Tip: You can usually type `cd ` and then drag-and-drop the unzipped folder from your file explorer directly into the terminal window to automatically paste the exact path.*

---

## Step 3: Set Up a Virtual Environment (Recommended)

While you *can* install things globally, it is best practice to create an isolated "virtual environment" for the project. This ensures the app's packages don't mess up other Python projects on your computer.

Inside your project folder, run the following commands:

**1. Create the environment:**

* **Windows/Mac/Linux:** ```bash
python -m venv venv
```


```



**2. Activate the environment:**

* **Windows (PowerShell):** ```powershell
.\venv\Scripts\Activate.ps1
```

```


* **Windows (Command Prompt):** ```cmd
.\venv\Scripts\activate.bat
```

```


* **Mac/Linux:** ```bash
source venv/bin/activate
```


```



You will know it worked because you will see `(venv)` appear at the very beginning of your terminal line.

---

## Step 4: Install the Required Packages

GitHub repositories usually include a file called `requirements.txt`. This file is just a list of all the Python packages (like Streamlit, Pandas, etc.) that the app needs to run.

To install everything on that list automatically, run:

```bash
pip install -r requirements.txt

```

Give it a minute or two to download and install everything.

---

## Step 5: Run the Streamlit App!

Now for the fun part. Look inside the folder to find the main Python file. It is usually named `app.py`, `main.py`, or `streamlit_app.py`.

Run the app using the `streamlit run` command followed by the filename:

```bash
streamlit run app.py

```

*(If the file is named something else, replace `app.py` with that name).*

### What happens next?

* Your terminal will spit out a couple of local URLs (usually `http://localhost:8501`).
* Your default web browser should **automatically open** a new tab displaying your newly running app.

Whenever you are done using the app, just go back to your terminal window and press **`Ctrl + C`** to stop the server and close the connection.

---



That's it! This app should make it easy and convenient to view SPSS files and convert them to Excel files. Feel free to modify the code to suit your needs, and please let me know if you have any issues or feedback.
