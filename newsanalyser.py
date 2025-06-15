import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import threading
import json
import re

from mcpnews.mcpnews import get_mcp_analysis

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Map bias to color (expand as needed)
BIAS_COLOR = {
    "Neutral": "green",
    "Slightly Negative": "orange",
    "Extreme Bias (Distraction)": "red",
    "None": "gray"
}

# Map country names to (lat, lon) for plotting
COUNTRY_COORDS = {
    "France": (46.603354, 1.888334),
    "United States": (37.09024, -95.712891),
    "USA": (37.09024, -95.712891),
    "Japan": (36.204824, 138.252924),
    "China": (35.86166, 104.195397),
    "India": (20.593684, 78.96288),
    "United Kingdom": (55.378051, -3.435973),
    "Germany": (51.165691, 10.451526),
    "Australia": (-25.274398, 133.775136),
    "Russia": (61.52401, 105.318756)
}

def get_country_from_newsoutlet(newsoutlet):
    # Extract country from "Outlet Name (Country)"
    match = re.search(r"\(([^)]+)\)", newsoutlet)
    if match:
        return match.group(1)
    return newsoutlet  # fallback

def plot_map(outlet_analysis):
    fig = plt.figure(figsize=(8, 5))
    m = Basemap(projection='robin', lon_0=0, resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='lightgray', lake_color='aqua')
    m.drawmapboundary(fill_color='aqua')

    plotted_countries = set()
    for entry in outlet_analysis:
        country = get_country_from_newsoutlet(entry.get("country_of_origin", ""))
        print("country is " +country)
        bias = entry.get("bias_level", "None")
        color = BIAS_COLOR.get(bias, "black")
        coords = COUNTRY_COORDS.get(country)
        if coords and country not in plotted_countries:
            x, y = m(coords[1], coords[0])
            m.plot(x, y, 'o', markersize=15, color=color, label=bias)
            plt.text(x, y, country, fontsize=9, ha='center', va='center', color='white', weight='bold')
            plotted_countries.add(country)

    # Custom legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10)
               for label, color in BIAS_COLOR.items()]
    plt.legend(handles=handles, loc='lower left', title="Bias Level")
    plt.title("Fairness/Bias of News Coverage by Country")
    return fig

def extract_outlet_analysis(llm_response):
    # Try to extract the JSON block from the LLM response
    try:
        # If the response is a JSON string, parse it directly
        if isinstance(llm_response, dict):
            return llm_response.get("outlet_analysis", [])
        # If the response is a string, try to find the JSON part
        code_block = re.search(r"```json(.*?)```", llm_response, re.DOTALL)
        if code_block:
            json_str = code_block.group(1).strip()
        else:
            start = llm_response.find('{')
            end = llm_response.rfind('}') + 1
            if start == -1 or end == -1:
                return []
            json_str = llm_response[start:end]
        data = json.loads(json_str)
        return data.get("articles", [])
    except Exception as e:
        print("Error parsing outlet_analysis:", e)
    return []

def show_map():
    llm_response = last_llm_response[0]
    print(llm_response)
    if not llm_response:
        messagebox.showinfo("No Data", "Please analyze a topic first.")
        return
    outlet_analysis = extract_outlet_analysis(llm_response)
    if not outlet_analysis:
        messagebox.showinfo("No Data", "No outlet_analysis found in LLM response.")
        return
    fig = plot_map(outlet_analysis)
    # Show in a new Tkinter window
    map_window = tk.Toplevel(root)
    map_window.title("News Fairness Map")
    canvas = FigureCanvasTkAgg(fig, master=map_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def analyze_topic():
    topic = topic_entry.get().strip()
    if not topic:
        messagebox.showwarning("Input Error", "Please enter a topic.")
        return
    analyze_button.config(state=tk.DISABLED)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "Analyzing, please wait...\n")

    def worker():
        result, error = get_mcp_analysis(topic)
        def update_gui():
            analyze_button.config(state=tk.NORMAL)
            result_text.delete(1.0, tk.END)
            if error:
                result_text.insert(tk.END, error)
                last_llm_response[0] = None
            else:
                result_text.insert(tk.END, "=== Prompt for LLM ===\n")
                result_text.insert(tk.END, result["prompt"] + "\n\n")
                result_text.insert(tk.END, "=== LLM Response ===\n")
                result_text.insert(tk.END, result["response"])
                last_llm_response[0] = result["response"]
        root.after(0, update_gui)
    threading.Thread(target=worker, daemon=True).start()

# Store last LLM response for map visualization
last_llm_response = [None]

root = tk.Tk()
root.title("Media Coverage Perspective Analyzer")

tk.Label(root, text="Enter topic to analyze:").pack(pady=(10,0))
topic_entry = tk.Entry(root, width=50)
topic_entry.pack(pady=5)
analyze_button = tk.Button(root, text="Analyze", command=analyze_topic)
analyze_button.pack(pady=5)

result_text = scrolledtext.ScrolledText(root, width=100, height=30, wrap=tk.WORD)
result_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

show_map_button = tk.Button(root, text="Show Map", command=show_map)
show_map_button.pack(pady=5)

root.mainloop()
