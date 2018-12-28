# staff-detect
Implementation of the paper bellow. Staff line detection can be used to get a grounding in OMR (Optical Music Recognition). It can handle computer generated scores aswell as most handwritten scores.

## Datasets and papers
[MUSCIMA++](https://ufal.mff.cuni.cz/muscima)

[DeepScores](https://tuggeluk.github.io/deepscores/)

```
@article{   su2012,
  author =  {Bolan Su , Shijian Lu , Umapada Pal and Chew Lim Tan},
  title =   {An Effective Staff Detection and Removal Technique for Musical Documents},
  journal = {2012 10th IAPR International Workshop on Document Analysis Systems},
  year =    {2012}
}
```

## Installation
(Requires OpenCV and Boost libraries).

1. `$ git clone ...`
2. `$ cd staff-detect && mkdir bin build `
3. `$ cd build && cmake ..`
4. `$ cd ../bin && ./stav --help`

There are many options to the `stav` binary. Just follow the instructions on the screen. For datascience/machine learning practitioners, it is possible to do batch detection of staves. All staves will be store in a convenient `xml` format. 

## Examples
The input image is scanned for staff line detection. The output is staff line positions. (ZOOM in to see staff identification)

<p align="center">  
<img src=pictures/computer_generated.png width="250" height="350">  
<img src=pictures/staff_computer_generated.png width="250" height="350">  
<img src=pictures/handwritten.png width="400" height="250">  
<img src=pictures/staff_handwritten.png width="400" height="250">  
</p>