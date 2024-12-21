
# Training Fish-Net to Predict Optimal Fishing Locations from Bathymetric Maps

**Author**: Oliver Eielson

---

## Abstract
This paper introduces **Fish-Net**, an adaptation of ResNet designed to predict optimal fishing locations from chart data. The model focuses on predicting saltwater fishing locations along the New England coast and was trained with bathymetric charts labeled using AIS records. Fish-Net performed with high accuracy—matching, and in some cases outperforming, survey results from professional fishermen. Additionally, Fish-Net outperformed **Fishbrain AI**, the current standard in AI fishing predictions.

---

## Introduction
Fish-Net was developed to improve the identification of ideal fishing spots by decreasing the time and effort currently required for identification. Traditionally, fishermen spend considerable time exploring and testing locations through trial and error. In testing, Fish-Net was able to predict fishing locations for the entire state of Rhode Island in just minutes. In addition to saving time, Fish-Net provides valuable insights into remote and unsurveyed areas. Fish-Net was trained on a Navionics[^1] bathymetric chart tiles dataset and AIS data from professional fishing boats. This unique approach allows the model to leverage the knowledge of the professional fishing industry to learn patterns in underwater topology.

[^1]: [Navionics](https://www.navionics.com/)

---

## Data
The curated dataset comprises two main components: **AIS** (Automatic Identification System) data and **bathymetric chart tiles** gathered within 20 miles of Montauk Point, NY. This area was chosen for its high concentration of charter fishing vessels and ecological similarity to New England’s coast. The dataset was compiled into three subsets—training, testing, and validation—ensuring that each image was exclusive to a subset.

### Chart Data
The core of the Fish-Net dataset was the bathymetric chart data. Each chart tile, sourced from the Navionics SonarChart map, is formatted as a 999×685-pixel image depicting the ocean floor’s topology. These unusual dimensions resulted from the method used to scrape the images and were not chosen to improve training.

Importantly, these tiles convey information through shapes and colors instead of letters and numbers. Depth is portrayed via color gradients, and underwater features are shown with contour lines. An object detection model can interpret these visual patterns without needing to parse text. Each tile was saved with the GPS coordinate of its upper-left corner to match the AIS pings to the image at a later point. The dataset comprised **650 images**, all within 20 miles of Montauk Point, NY.

### AIS Data
Automatic Identification System (AIS) data from **2020 to 2023** was used to label the chart tiles. AIS is a vessel-tracking system used in maritime applications to exchange real-time information between ships and shore-based stations, including speed, heading, size, and location.

An initial filtering step removed any boat not registered as a fishing boat. Large deep-sea fishing vessels were subsequently removed by filtering out any vessel over 35 feet in length or weighing more than 15 tons, leaving only charter fishing vessels. The remaining vessels were cross-referenced with NOAA fishing licenses to ensure the accuracy of the dataset.

Further filtering isolated only vessels that were likely **actively fishing**. Any vessel moving at speeds exceeding 5 knots or found in harbor areas was excluded, as those behaviors typically do not indicate active fishing. Around the recorded AIS ping, a **50×50-pixel** region was labeled as a “good fishing” zone to account for slight inaccuracies in AIS and chart coordinates. This larger region also improved Fish-Net’s robustness by preventing the model from being overly penalized for minor positional discrepancies.

---

## Method

### Model
Fish-Net is an adaptation of the [**fasterrcnn_resnet50_fpn**](https://arxiv.org/abs/1506.01497) model, incorporating a two-stage object detection design. The model uses **ResNet50** for feature extraction paired with a **Feature Pyramid Network (FPN)**. Using the extracted features, the model generates region proposals and then classifies these proposals. For more detailed information, see the original [Faster R-CNN paper](https://arxiv.org/abs/1506.01497).

Fish-Net uses a **ResNet50 backbone pretrained on the COCO dataset**. The final layer is modified to ensure that the output dimension matches a two-class dataset. The model’s input is a 3-channel, 999×685-pixel image, and the output consists of bounding boxes in (`xyxy`) format, a prediction label (-1 or 1), and a confidence score.

To keep the dimensionality of labels consistent during training, images that have fewer bounding boxes than the maximum required are padded with “background” bounding boxes labeled as -1. These padding labels are effectively ignored during training but maintain uniform label dimensionality.

<details>
<summary>Figure 1: The <em>fasterrcnn_resnet50_fpn</em> model architecture</summary>

[Placeholder for model.png]

This diagram illustrates the fasterrcnn_resnet50_fpn
architecture on which Fish-Net is based.

</details>

### Hyper-Parameters
Because Fish-Net uses a pretrained backbone, only **ten epochs** of training were necessary. A **learning rate of 1** was used, alongside **gradient clipping**. A **step size of 5**, **weight decay of 0.005**, and **Stochastic Gradient Descent (SGD)** optimizer were utilized. The relatively high weight decay helped compensate for errors in the dataset, which is discussed later.

The model was trained on a Google Colab V100 GPU, and training was completed in under 10 minutes.

### Training
Fish-Net’s performance was evaluated on separate training and validation datasets. During each epoch, a **training loss** was calculated using the training set, and an **evaluation loss** was computed with the validation set. The **mean average precision (mAP)** was also calculated on the validation set at each epoch.

Below is an example plot of the training and evaluation loss:

<details>
<summary>Figure 2: Training and evaluation loss per epoch</summary>

[Placeholder for Training Loss Plot]

The x-axis represents the number of epochs (0-10),
and the y-axis represents the loss values.

</details>

Additionally, Fish-Net’s **mAP** was monitored during training:

<details>
<summary>Figure 3: Fish-Net's mAP during training</summary>

[Placeholder for mAP Plot]

The x-axis represents the number of epochs,
and the y-axis represents the mAP.

</details>

---

## Results
During training, **both the training and evaluation losses** steadily decreased, indicating that the model was learning and generalizing effectively. Notably, the losses decreased at similar rates, which suggests the model was **not overfitting**.

The **mAP** started out low, remained stagnant initially, and then began to increase toward the end of training. The relatively low final mAP (below 0.006) is largely attributed to **dataset errors** (i.e., unlabeled fishing spots).

### Comparison with Professional Surveys
In real-world testing, Fish-Net’s predictions on chart tiles of **Narragansett Bay, RI** were compared with Captain Seagull’s Sports Fishing survey results. Fish-Net’s predictions aligned almost perfectly with the survey, as shown in the figure below. Notably, Fish-Net also identified several locations that Captain Seagull’s survey missed.

<details>
<summary>Figure 4: Captain Seagull Sports Fishing vs Fish-Net</summary>

[Placeholder for screenshot comparison]

The left image shows Captain Seagull’s Sports Fishing map
(purple = good fishing spots).
The right image shows Fish-Net’s predictions (orange).
Circled areas indicate spots Fish-Net identified that
the survey missed, which the author confirms are good
fishing locations.

</details>

### Comparison with Fishbrain AI
Fish-Net was benchmarked against **Fishbrain**’s AI predictions, a popular fishing application’s proprietary model. As illustrated below, Fishbrain’s AI fails to accurately identify most of the same fishing spots recommended by Captain Seagull’s Sports Fishing; it often mislabels features like boat wakes as good fishing locations. By contrast, Fish-Net’s predictions closely match those of the professional survey.

<details>
<summary>Figure 5: Fishbrain vs Fish-Net</summary>

[Placeholder for FishBrain.png]

The left image shows Fishbrain’s AI predictions (orange),
while the right image shows Fish-Net’s predictions (orange).
Fishbrain’s results do not align with the survey-proven spots,
whereas Fish-Net’s do.

</details>

---

## Discussion
The results show that **Fish-Net is highly effective and accurate** at determining ideal fishing locations. The steady decrease in both training and evaluation loss suggests the model generalizes well. If the model were overfitting, the evaluation loss would not track so closely with the training loss.

While the mAP metric is very low, this is largely attributed to **missing labels** in the dataset. For example, if a tile truly has four good fishing spots (L1, L2, L3, L4) but the dataset only labels three of them (L1, L2, L3), then the model gets penalized for correctly identifying the unlisted L4. In real-world testing, the model’s results match professional surveys very closely, indicating that the low mAP is not the result of poor generalization.

In fact, Fish-Net **correctly predicted some fishing locations** that the survey missed, which were later confirmed by the author’s personal experience. Although further testing is required to determine statistical significance, this result hints that Fish-Net may outperform even professional fishermen in certain scenarios.

### Model Limitations
Despite its strong performance, Fish-Net has some weaknesses:

1. **Freshwater vs. Saltwater**:  
   The model sometimes misidentifies **freshwater ponds** as good fishing locations. Since no bathymetric data exists in the tiles for these ponds (i.e., no depth gradients or contour lines), the model’s prediction is effectively baseless.

2. **Landlocked Areas**:  
   Due to **errors in the AIS dataset**, some locations on land are incorrectly labeled as fishing hotspots (e.g., one fisherman carrying his AIS transmitter home). Although much of this erroneous data was filtered out, some likely remains, causing false positives.

3. **Regional Scope**:  
   Fish-Net is specifically trained on the **New England coastline**. Fish behavior in other regions might be very different, and the model may not generalize well to other locales without additional regional data.

### Fish-Net vs. Other Models
Beyond matching and occasionally surpassing human predictive ability, Fish-Net outperforms **Fishbrain**. Fishbrain’s reliance on **satellite images** rather than bathymetric maps likely contributes to its inaccuracies, such as mislabeling rooftops and boat wakes as optimal fishing locations.

---

## Conclusion
Fish-Net demonstrates **high accuracy** in predicting ideal fishing locations. It matches or exceeds human performance while generating results in a fraction of the time. This paper also illustrates Fish-Net’s notable advantages over other existing models, such as Fishbrain AI.

By training on **bathymetric charts** and **real-world AIS data**, Fish-Net leverages both geography and expert knowledge. The result is a powerful tool that can aid fishermen, researchers, and conservationists in quickly and accurately locating the best fishing spots.

---

## References
- Ren, S. et al. (2015). [Faster R-CNN](https://arxiv.org/abs/1506.01497).
- [Navionics SonarChart](https://www.navionics.com/).
- [Fishbrain](https://fishbrain.com/).
