# mk_animation (#11)

Text scrolling animation that displays a text file and saves as AVI.

## Approach

- Reads a text file and wraps long lines to the specified character width.
- Uses matplotlib's `FuncAnimation` to create a scrolling animation.
- Saves the animation using FFMpeg if available, with fallbacks to Pillow or OpenCV.
- Supports configurable lines per frame, scroll speed, FPS, and duration.
