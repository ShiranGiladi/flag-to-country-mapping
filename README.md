# Flag to Country Mapping System 🌍🏁

An interactive computer vision-powered geography application that combines hand gesture recognition with real-time world map visualization. Select country flags using intuitive hand gestures and watch as their locations are highlighted on a live world map display.

## ✨ Features

- **Real-time Hand Gesture Recognition**: Uses MediaPipe for accurate hand tracking and gesture detection
- **Interactive Flag Grid**: Browse through a scrollable grid of country flags
- **Live World Map Visualization**: See countries highlighted on an actual world map
- **Gesture-Based Navigation**: 
  - Pinch gesture (thumb + index finger) to select countries
  - Two-finger scroll for navigation through the flag grid
- **Geographic Information Display**: Shows continent information for selected countries
- **Visual Feedback**: Pulsing dots and highlighting for selected countries
- **Responsive UI**: Clean, organized interface with real-time FPS counter

## 🎮 How to Use

### Hand Gestures
1. **Country Selection**: Make a pinch gesture (bring thumb and index finger together) while pointing at a flag
2. **Scroll Navigation**: Extend index and middle fingers, move up/down to scroll through flag pages
3. **Exit**: Press 'q' key to quit the application

### Visual Interface
- **Left Panel**: Scrollable grid of country flags with names
- **Right Panel**: Interactive world map with country highlighting
- **Bottom**: Current selection info and continent details

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- Webcam/Camera
- Required data files (world map, flag images, geographic data)

### Required Packages
```bash
pip install opencv-python
pip install mediapipe
pip install pandas
pip install numpy
pip install python-dotenv
```

### Setup Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ShiranGiladi/flag-to-country-mapping.git
   cd flag-to-country-mapping
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Required Data Files**
   You'll need the following files:
   - `world_map.jpg` - World map image
   - `countries.geo.json` - Geographic data for countries
   - `countries_data.csv` - Country to continent mapping
   - `flags/` directory with country flag images (PNG format)

4. **Configure Environment Variables**
   Create a `.env` file in the root directory:
   ```env
   WORLD_MAP_PATH=your_path_here/world_map.jpg
   GEO_DATA_PATH=your_path_here/countries.geo.json
   FLAGS_DIR=your_path_here/flags/
   COUNTRIES_CSV_PATH=your_path_here/countries.csv
   ```

5. **Prepare Flag Images**
   - Place flag images in the `flags/` directory
   - Name files exactly as country names (e.g., `Israel.png`, `Vietnam.png`)
   - Supported format: PNG

6. **Run the Application**
   ```bash
   python main.py
   ```

## 🔧 Configuration

### Environment Variables
- `WORLD_MAP_PATH`: Path to world map image
- `GEO_DATA_PATH`: Path to countries GeoJSON file
- `FLAGS_DIR`: Directory containing flag images
- `COUNTRIES_CSV_PATH`: Path to countries CSV file

### Customizable Parameters
- Grid dimensions and layout
- Gesture sensitivity thresholds
- Scroll and click cooldown periods
- Map scaling and positioning

## 📊 Technical Details

### Technologies Used
- **OpenCV**: Computer vision and image processing
- **MediaPipe**: Hand landmark detection and tracking
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and CSV handling
- **JSON**: Geographic data processing

### Key Components
1. **FlagToCountrySystem**: Main application class
2. **Hand Gesture Detection**: Real-time finger position tracking
3. **Geographic Coordinate Mapping**: Conversion from lat/lon to pixel coordinates
4. **UI Rendering**: Flag grid and world map visualization
5. **Data Management**: Country information and flag image caching
