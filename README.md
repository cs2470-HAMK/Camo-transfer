# Camo-transfer

The goal here is to build out a model that is able to take in any landscape image and output a reasonable camouflage pattern that a potential user could apply to tasks and projects of interests. Much of the work is parceled into separate colab notebooks that collectively produce a model output. 

### Data

Image data is divided into two primary data sets: camouflage patterns and natural landscapes. Camouflage images were scraped from an online camouflage pattern database. Specifically, these images are standardized as 17 KB cutouts of camouflage patterns with a watermark of the database in the middle. In total, 2281 patterns were successfully scraped from the database. Landscape images come from the Landscapes High Quality dataset. This data consists of 90,000 various landscape images comprising both natural and man-made features. To conserve computational resources, these images were resampled and then processed to greatly reduce file sizes. The images are resampled utilizing Python’s pillow library and standardized to match a shape of 256x256 pixels. 


### Preprocessing 

After extracting all images, distinct preprocessing techniques were applied to both image sets to aid model training. For camouflage images, watermarks were removed to prevent them from appearing in our style-transfer outputs. Intuitively, the inclusion of a watermark in a landscape image would significantly reduce model efficacy. For landscape images, sky-based features were removed during preprocessing. By removing the sky features, again, irrelevant image-features were excluded from model outputs. Finally, all images were standardized to the same size and data type. After, the data was split into training and testing sets for model initialization. 

### Architecture	
The first baseline model used was an off-the-shelf CycleGAN architecture. Given the context of the problem, this was expected to perform poorly because the role of the second generator, as well as the cycle-consistency loss, achieves the opposite intended effect. The original landscape ideally should not be reconstructible from the generated camouflage. While the ‘coherence’ of the original landscape should not be transferred, the colors of the landscape should still be kept. To address these concerns iterative changes were made to the model architecture in several ways. 
First, the camo-landscape generator was removed and patching was implemented. Applying evaluating camouflage generation based on its coherence with a subset of the landscape imagery ensured that smaller textures of the landscape were learned instead of higher-level outlines (e.g, mountains, horizon, etc.). 

The landscape discriminator instead learned to identify which landscape is real between the original landscape and the one patched with the generated camouflage. This in turn encourages the camouflage generator to produce camouflages that are difficult to discern from the surrounding patterns in the input landscape. 

Second, in order to specifically target the transfer of the distribution of colors from the landscape to the corresponding camouflage, an additional penalty was added to the generator loss: for each training landscape, the five largest groupings of colors were determined through K-means clustering for both the landscape and the generated camouflage. The generator loss is penalized by the distance between the two distributions of colors, which further motivates the generator to produce camouflages that match the general proportion of colors in the input landscapes. 

The overall architecture of the model can be visualized below. Where the model takes a landscape image, which is fed into the generator. This then produces a camouflage rendering that represents the most descriptive concealment pattern from additional camouflage samples. Provided with a discriminator for both the generated image itself and a patched version, the outputs are evaluated on a custom loss function that prioritizes the goals of the project (e.g, color distribution, quality of concealment within earth portions of landscape). 
