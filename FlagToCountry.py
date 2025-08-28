import cv2
import handTrackingModule as htm
import time
import pandas as pd
import numpy as np
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FlagToCountrySystem:
    def __init__(self):
        self.detector = htm.handDetector(maxHands=1, detectionCon=0.7)
        self.cap = cv2.VideoCapture(0)
        
        # Load paths from environment variables 
        self.world_map_path = os.getenv('WORLD_MAP_PATH')
        self.geo_data_path = os.getenv('GEO_DATA_PATH')
        self.flags_dir = os.getenv('FLAGS_DIR')
        self.countries_csv_path = os.getenv('COUNTRIES_CSV_PATH')
        
        # Validate that all required paths are set
        if not all([self.world_map_path, self.geo_data_path, self.flags_dir, self.countries_csv_path]):
            raise ValueError("Missing required environment variables. Please check your .env file.")
        
        # Grid UI Configuration
        self.grid_x = 50
        self.grid_width = 500 
        self.grid_height = 550  
        self.flag_width = 75   
        self.flag_height = 50  
        self.grid_cols = 5  
        self.grid_rows = 7  
        self.cell_width = self.grid_width // self.grid_cols
        self.cell_height = 85  
        self.scroll_offset = 0
        self.selected_country = None
        
        # Gesture thresholds
        self.click_threshold = 60  # Distance threshold for pinch gesture
        self.scroll_cooldown = 0
        self.click_cooldown = 0
        
        # Load data and flags
        self.countries_data = self.load_countries_geo_data()
        self.flag_images = {}   # Cache for flag images
        self.world_map = self.load_world_map()
        
        # Map configuration
        self.map_x = 570  # X position where map starts (moved right to accommodate wider grid)
        self.map_y = 50   # Y position where map starts
        self.map_scale = 1.0
        
    def load_world_map(self):
        """Load the world map image"""
        try:
            if os.path.exists(self.world_map_path):
                map_img = cv2.imread(self.world_map_path)
                # Resize map to fit in the available space
                target_width, target_height = 900, 650
                map_img = cv2.resize(map_img, (target_width, target_height))
                return map_img
            else:
                print(f"World map file not found: {self.world_map_path}")
                # Create a placeholder map
                placeholder = np.zeros((600, 800, 3), dtype=np.uint8)
                cv2.rectangle(placeholder, (0, 0), (800, 600), (100, 100, 100), -1)
                cv2.putText(placeholder, "World Map Not Found", (250, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return placeholder
        except Exception as e:
            print(f"Error loading world map: {e}")
            placeholder = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.rectangle(placeholder, (0, 0), (800, 600), (200, 0, 0), -1)
            return placeholder
        
    def load_countries_geo_data(self):
        """
        Load countries data from countries.geo.json file, but only include countries that have flag images 
        (meaning include the countries that in the flags folder)
        Flag files should be named exactly as the country name (e.g., 'Israel.png', 'Vietnam.png')
        """
        try:
            # Load geo data
            if not os.path.exists(self.geo_data_path):
                print(f"Geo data file not found: {self.geo_data_path}")
                return []
            
            with open(self.geo_data_path, 'r', encoding='utf-8') as f:
                geo_data = json.load(f)
            
            # Get list of available flag files
            if not os.path.exists(self.flags_dir):
                print(f"Flags directory not found: {self.flags_dir}")
                return []
            
            flag_files = [f for f in os.listdir(self.flags_dir) if f.endswith('.png')]
            available_countries = [os.path.splitext(f)[0] for f in flag_files]
            
            print(f"Found {len(flag_files)} flag files")
            
            # Filter geo data to only include countries with flag images
            countries_with_flags = []
            
            for feature in geo_data['features']:
                country_name = feature['properties']['name']
                
                # Check if we have a flag for this country
                if country_name in available_countries:
                    # Calculate centroid for placing the dot
                    centroid = self.calculate_country_centroid(feature['geometry'])
                    
                    countries_with_flags.append({
                        'name': country_name,
                        'flag_file': f"{country_name}.png",
                        'geometry': feature['geometry'],
                        'centroid': centroid,
                        'continent': self.get_continent(country_name) if pd.notna(self.get_continent(country_name)) else '-'
                    })
            
            print(f"Loaded {len(countries_with_flags)} countries with flags")
            return sorted(countries_with_flags, key=lambda x: x['name'])  # Sort alphabetically
            
        except Exception as e:
            print(f"Error loading countries geo data: {e}")
            return []
    
    def calculate_country_centroid(self, geometry):
        """Calculate the centroid (center point) of a country's geometry"""
        try:
            if geometry['type'] == 'Polygon':
                coordinates = geometry['coordinates'][0]  # Take outer ring
            elif geometry['type'] == 'MultiPolygon':
                # For multipolygon, take the largest polygon
                largest_polygon = max(geometry['coordinates'], key=lambda x: len(x[0]))
                coordinates = largest_polygon[0]
            else:
                return (0, 0)
            
            # Calculate centroid
            x_coords = [coord[0] for coord in coordinates]
            y_coords = [coord[1] for coord in coordinates]
            
            centroid_x = sum(x_coords) / len(x_coords)
            centroid_y = sum(y_coords) / len(y_coords)
            
            return (centroid_x, centroid_y)
        except Exception as e:
            print(f"Error calculating centroid: {e}")
            return (0, 0)
    
    def get_continent(self, country_name):
        """Get continent for a country from CSV file"""
        try:
            # Load the CSV file if not already loaded
            if not hasattr(self, 'continent_data'):
                self.continent_data = pd.read_csv(self.countries_csv_path)
                
                # Clean up column names and country names (strip whitespace)
                self.continent_data['Country'] = self.continent_data['Country'].str.strip()
                self.continent_data['Continent'] = self.continent_data['Continent'].str.strip()
                
                # Create a dictionary for faster lookup
                self.continent_lookup = dict(zip(self.continent_data['Country'], self.continent_data['Continent']))
            
            # Look up the continent for the given country
            continent = self.continent_lookup.get(country_name, 'Unknown')
            return continent
            
        except Exception as e:
            print(f"Error loading continent data from CSV: {e}")            
            return 'Unknown'
    
    def load_flag_image(self, flag_file):
        """Load and cache flag image"""
        if flag_file in self.flag_images:
            return self.flag_images[flag_file]
        
        try:
            flag_path = os.path.join(self.flags_dir, flag_file)
            if os.path.exists(flag_path):
                flag_img = cv2.imread(flag_path)
                # Resize flag to fit in the grid cell
                flag_img = self.resize_flag(flag_img, self.flag_width, self.flag_height)
                self.flag_images[flag_file] = flag_img
                return flag_img
            else:
                # Create placeholder flag
                placeholder = np.zeros((self.flag_height, self.flag_width, 3), dtype=np.uint8)
                cv2.rectangle(placeholder, (0, 0), (self.flag_width, self.flag_height), (100, 100, 100), -1)
                cv2.putText(placeholder, "N/A", (15, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                self.flag_images[flag_file] = placeholder
                return placeholder
        except Exception as e:
            print(f"Error loading flag {flag_file}: {e}")
            placeholder = np.zeros((self.flag_height, self.flag_width, 3), dtype=np.uint8)
            cv2.rectangle(placeholder, (0, 0), (self.flag_width, self.flag_height), (200, 0, 0), -1)
            self.flag_images[flag_file] = placeholder
            return placeholder
    
    def resize_flag(self, flag_img, target_width, target_height):
        """Resize flag image maintaining aspect ratio"""
        h, w = flag_img.shape[:2]
        
        # Calculate scaling factor
        scale = min(target_width / w, target_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(flag_img, (new_w, new_h))
        
        # Create canvas with target size
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Center the resized image on canvas
        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    def draw_world_map(self, img):
        """Draw the world map on the right side of the screen"""
        if self.world_map is not None:
            map_height, map_width = self.world_map.shape[:2]
            img_height, img_width = img.shape[:2]
            
            # Calculate position to place the map (right side)
            start_x = self.map_x
            start_y = self.map_y
            
            # Make sure the map fits within the image bounds
            end_x = min(start_x + map_width, img_width)
            end_y = min(start_y + map_height, img_height)
            
            # Adjust map size if it doesn't fit
            actual_width = end_x - start_x
            actual_height = end_y - start_y
            
            if actual_width < map_width or actual_height < map_height:
                # Resize map to fit
                resized_map = cv2.resize(self.world_map, (actual_width, actual_height))
                img[start_y:end_y, start_x:end_x] = resized_map
                # Update map dimensions for dot placement
                self.map_display_x = start_x
                self.map_display_y = start_y
                self.map_display_width = actual_width
                self.map_display_height = actual_height
            else:
                img[start_y:start_y + map_height, start_x:start_x + map_width] = self.world_map
                self.map_display_x = start_x
                self.map_display_y = start_y
                self.map_display_width = map_width
                self.map_display_height = map_height
    
    def draw_flag_grid(self, img, visible_countries):
        """Draw the flag grid on the left side"""
        # Create a copy of the original image for overlay
        overlay = img.copy()

        # Draw background rectangle
        cv2.rectangle(overlay, (self.grid_x - 10, 50), 
                     (self.grid_x + self.grid_width + 10, 50 + self.grid_height + 100), 
                     (50, 50, 50), -1)

        # Blend the overlay with the original image
        alpha = 0.3  # Transparency factor
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # Draw title
        cv2.putText(img, f"Countries ({len(self.countries_data)})", (self.grid_x, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw scroll indicators
        if self.scroll_offset > 0:
            cv2.putText(img, "^ Scroll Up", (self.grid_x + 10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        total_rows = (len(self.countries_data) + self.grid_cols - 1) // self.grid_cols
        if self.scroll_offset + self.grid_rows < total_rows:
            cv2.putText(img, "v Scroll Down", (self.grid_x + 10, 50 + self.grid_height + 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw grid of flags
        for i, country in enumerate(visible_countries):
            row = i // self.grid_cols
            col = i % self.grid_cols
            
            # Skip if we've exceeded the visible rows
            if row >= self.grid_rows:
                break
            
            # Calculate position
            cell_x = self.grid_x + col * self.cell_width
            cell_y = 80 + row * self.cell_height
            
            # Highlight selected country
            if self.selected_country and self.selected_country['name'] == country['name']:
                cv2.rectangle(img, (cell_x - 2, cell_y - 2), 
                             (cell_x + self.cell_width + 2, cell_y + self.cell_height + 2), 
                             (0, 255, 0), 3)
            
            # Draw cell border with padding
            cell_padding = 8
            cv2.rectangle(img, (cell_x + cell_padding, cell_y + cell_padding), 
                         (cell_x + self.cell_width - cell_padding, cell_y + self.cell_height - cell_padding), 
                         (100, 100, 100), 1)
            
            # Draw country flag (centered in cell)
            flag_img = self.load_flag_image(country['flag_file'])
            flag_x = cell_x + (self.cell_width - self.flag_width) // 2
            flag_y = cell_y + 10  # More padding from top
            
            # Ensure flag fits within bounds
            if (flag_y + flag_img.shape[0] < img.shape[0] and 
                flag_x + flag_img.shape[1] < img.shape[1] and
                flag_x >= 0 and flag_y >= 0):
                img[flag_y:flag_y + flag_img.shape[0], 
                    flag_x:flag_x + flag_img.shape[1]] = flag_img
            
            # Draw country name (truncated if too long)
            country_name = country['name']
            if len(country_name) > 10:
                country_name = country_name[:10] + "..."
            
            text_x = cell_x + 10
            text_y = cell_y + self.flag_height + 23
            cv2.putText(img, country_name, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
    def highlight_country_by_chosen_flag(self, img):
        """Draw dot within the selected country's coordinates on the world map"""
        if self.selected_country and hasattr(self, 'map_display_x'):
            try:
                # Get country centroid (longitude, latitude)
                lon, lat = self.selected_country['centroid']
                
                # Convert geographical coordinates to map pixel coordinates
                # Assuming the world map uses standard projection (longitude: -180 to 180, latitude: -90 to 90)
                # Map longitude (-180 to 180) to pixel x (0 to map_width)
                pixel_x = int(((lon + 180) / 360) * self.map_display_width)
                # Map latitude (90 to -90) to pixel y (0 to map_height) - note: y increases downward
                pixel_y = int(((90 - lat) / 180) * self.map_display_height)
                
                # Adjust to map position on screen
                dot_x = self.map_display_x + pixel_x
                dot_y = self.map_display_y + pixel_y
                
                # Make sure the dot is within the map bounds
                if (self.map_display_x <= dot_x <= self.map_display_x + self.map_display_width and 
                    self.map_display_y <= dot_y <= self.map_display_y + self.map_display_height):
                    
                    # Draw a pulsing dot
                    pulse = int(abs(np.sin(time.time() * 3)) * 10) + 5  # Pulse between 5 and 15
                    cv2.circle(img, (dot_x, dot_y), pulse, (0, 0, 255), -1)  # Red filled circle
                    cv2.circle(img, (dot_x, dot_y), pulse + 5, (255, 255, 255), 2)  # White border
                    
                    # Draw country name near the dot
                    text_x = dot_x + 20
                    text_y = dot_y - 10
                    cv2.putText(img, self.selected_country['name'], (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Draw continent info below the map
                    info_text = f"Selected: {self.selected_country['name']} ({self.selected_country['continent']})"
                    cv2.putText(img, info_text, (self.map_display_x, self.map_display_y + self.map_display_height + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                               
            except Exception as e:
                print(f"Error highlighting country: {e}")
    
    def get_visible_countries(self):
        """Get countries that should be visible based on scroll offset"""
        countries_per_page = self.grid_cols * self.grid_rows
        start_idx = self.scroll_offset * self.grid_cols
        end_idx = min(len(self.countries_data), start_idx + countries_per_page)
        return self.countries_data[start_idx:end_idx], start_idx
    
    def check_country_click(self, lmList, visible_countries, start_idx):
        """Check if user clicked on a country in the grid"""
        if len(lmList) == 0:
            return None
        
        # Get index finger position (landmark 8)
        finger_x, finger_y = lmList[8][1], lmList[8][2]
        
        # Check if click is in grid area
        if (self.grid_x <= finger_x <= self.grid_x + self.grid_width and
            80 <= finger_y <= 80 + self.grid_rows * self.cell_height):
            
            # Calculate which grid cell was clicked
            rel_x = finger_x - self.grid_x
            rel_y = finger_y - 80
            
            col = rel_x // self.cell_width
            row = rel_y // self.cell_height
            
            # Calculate index in visible countries list
            grid_index = row * self.grid_cols + col
            
            # Check if the index is valid
            if 0 <= grid_index < len(visible_countries):
                return visible_countries[grid_index]
        
        return None
    
    def run(self):
        """Main execution loop"""
        if not self.countries_data:
            print("No countries with flags found! Please check your 'flags' directory and 'countries.geo.json' file.")
            return
        
        pTime = 0
        total_rows = (len(self.countries_data) + self.grid_cols - 1) // self.grid_cols
        max_scroll_rows = max(0, total_rows - self.grid_rows)
        
        print(f"Starting system with {len(self.countries_data)} countries loaded.")
        print("Controls:")
        print("- Index + Middle fingers up: Scroll through pages")
        print("- Thumb + Index pinch: Select country")
        print("- Press 'q' to quit")

        while True:
            success, img = self.cap.read()
            if not success:
                break
                
            img = cv2.flip(img, 1)  # Mirror image for better user experience
            img = self.detector.findHands(img)
            lmList = self.detector.findPosition(img, draw=False)
            
            # Draw world map first
            self.draw_world_map(img)
            
            # Get visible countries
            visible_countries, start_idx = self.get_visible_countries()

            if len(lmList) != 0:
                fingers = self.detector.fingersUp()
                current_time = time.time()
                
                distance, _, pointsInfo = self.detector.findDistance(4, 8, img, draw=True)
                
                # Clicking Mode: Thumb and Index close together
                if distance < self.click_threshold:
                    cv2.circle(img, (pointsInfo[4], pointsInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    if current_time - self.click_cooldown > 0.8:  # Cooldown to prevent multiple clicks
                        clicked_country = self.check_country_click(lmList, visible_countries, start_idx)
                        if clicked_country:
                            self.selected_country = clicked_country
                            self.click_cooldown = current_time
                            print(f"Selected: {clicked_country['name']} (Continent: {clicked_country['continent']})")

                # Scrolling Mode: Index and Middle fingers up
                elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
                    if current_time - self.scroll_cooldown > 0.5:  # Cooldown to prevent rapid scrolling
                        # Get middle finger position for scroll direction
                        middle_finger_y = lmList[12][2]  # Middle finger tip
                        img_center_y = img.shape[0] // 2
                        
                        x1, y1 = lmList[8][1:]
                        x2, y2 = lmList[12][1:]
                        cv2.rectangle(img, (x1, y1-25), (x2, y2+25), (0, 255, 0), cv2.FILLED)

                        if middle_finger_y < img_center_y - 50:  # Scroll up
                            self.scroll_offset = max(0, self.scroll_offset - 1)
                            self.scroll_cooldown = current_time
                        elif middle_finger_y > img_center_y:  # Scroll down
                            self.scroll_offset = min(max_scroll_rows, self.scroll_offset + 1)
                            self.scroll_cooldown = current_time
            
            # Draw UI elements
            self.draw_flag_grid(img, visible_countries)
            self.highlight_country_by_chosen_flag(img)
            
            # FPS Counter
            cTime = time.time()
            fps = 1 / (cTime - pTime) if pTime != 0 else 0
            pTime = cTime
            cv2.putText(img, f'FPS: {int(fps)}', (20, img.shape[0] - 280), 
                       cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            
            # Instructions
            cv2.putText(img, "Grid Navigation: Index+Middle | Select: Thumb+Index Pinch", 
                       (20, img.shape[0] - 300), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
            
            cv2.imshow("Flag to Country Mapping System", img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    try:
        system = FlagToCountrySystem()
        system.run()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("Please make sure your .env file exists and contains all required paths.")
    except Exception as e:
        print(f"Error starting system: {e}")

if __name__ == "__main__":
    main()