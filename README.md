# COLLAB_DRAW

A Kivy-based Python implementation of a drawing app that allows you and an AI to draw strokes interchangeably. 

## Default Parameters

Using default parameters, the app will track every stroke drawn by the user. When the user lifts their pen/mouse, the app feeds the *last drawn stroke* to the pre-loaded AI. The app will then draw out the subsequent stroke predicted by the AI in the same style.

## Custom Parameters
Exchange parameters allow the user to modify how the computer and human artists interact such as in terms of:

- rate of exchange (number of strokes from human vs. computer)
- AI memory (how many  prior strokes the AI considers when deciding the next stroke)
- generation type (encode, encode-decode, random generation — these determine how the next stroke is generated)
..- decode: the AI encodes your drawn strokes and predicts the next stroke
..- encode-decode: the AI encodes the drawn image so far (a pixel-based png) and predicts the next stroke — greater spatial understanding
..- random generation

Performance parameters modify how the UI of the application:

- computer stroke display (the computer can simulate a slow stroke movement like a human hand or display its output all at once or withhold its output until the end of the session)
- undo option
- user color choice option

## Pen Textures

Textures can be loaded in by placing textures in a 'textures' folder named 'brush#.jpg' where # is 0,1,2,3 etc. 
