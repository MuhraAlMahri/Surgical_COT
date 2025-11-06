# PRESENTATION EXAMPLES - SURGICAL VQA
# Selected from Exp4 (Curriculum Learning - Best Performer)

================================================================================

## ✅ STAGE 1 SUCCESS EXAMPLES (Quality Control & Initial Assessment)


### Example 1:
**Question:** Is there a green/black box artefact?
**Ground Truth:** yes
**Model Says:** Yes, there is a green/black box artefact in the image.
**Extracted Answer:** yes ✓


### Example 2:
**Question:** How many polyps are in the image?
**Ground Truth:** 1
**Model Says:** There is one polyp in the image.
**Extracted Answer:** 1 ✓


### Example 3:
**Question:** Is this finding easy to detect?
**Ground Truth:** no
**Model Says:** The image appears to show a dark mass or lesion inside a body cavity, possibly the esophagus or another digestive tract. This type of finding can be d...
**Extracted Answer:** no ✓


### Example 4:
**Question:** Is there text?
**Ground Truth:** yes
**Model Says:** Yes, there is text in the image. The text includes:

- Date: 17/10/2012
- Time: 16:58:53
- CVP: 2
- Grade: N
- Biopsy: A5
**Extracted Answer:** yes ✓


### Example 5:
**Question:** Is there text?
**Ground Truth:** yes
**Model Says:** Yes, there is text in the image. The text includes:

- "Video 576i"
- "4.25 Mbit/s"
- "1 Drop"
- "0 Buf"
- "21/08/2013"
- "09:45:16"
- "CVP:5"
- "Gr:N...
**Extracted Answer:** yes ✓


================================================================================

## ✅ STAGE 2 SUCCESS EXAMPLES (Findings & Anatomical Identification)


### Example 1:
**Question:** Have all polyps been removed?
**Ground Truth:** no
**Model Says:** The image you provided appears to be a medical endoscopic image, likely of the colon or another internal organ. It shows a polyp-like structure. To de...
**Extracted Answer:** no ✓


### Example 2:
**Question:** Does this image contain any finding?
**Ground Truth:** no
**Model Says:** This image appears to be an endoscopic view of the gastrointestinal tract, likely the colon or small intestine. The image shows the inner lining of th...
**Extracted Answer:** no ✓


### Example 3:
**Question:** How many findings are present?
**Ground Truth:** 1
**Model Says:** There are two findings present in the image:

1. A large, elevated lesion with a yellowish appearance.
2. A smaller, more circular lesion with a simil...
**Extracted Answer:** 1 ✓


### Example 4:
**Question:** How many findings are present?
**Ground Truth:** 1
**Model Says:** The image appears to show an endoscopic view of the gastrointestinal tract. There are several white, raised lesions visible on the mucosal surface. Ba...
**Extracted Answer:** 1 ✓


### Example 5:
**Question:** Does this image contain any finding?
**Ground Truth:** no
**Model Says:** The image appears to be an endoscopic view of the interior of a hollow organ, likely the esophagus or another digestive tract. The main image shows a ...
**Extracted Answer:** no ✓


================================================================================

## ❌ STAGE 1 FAILURE EXAMPLES (Learning Opportunities)


### Example 1:
**Question:** Where in the image is the abnormality?
**Ground Truth:** center; upper-center
**Model Says:** The abnormality in the image appears to be a polyp-like structure located in the center of the image. This structure is elevated and has a distinct ap...
**Extracted Answer:** no ✗
**Problem:** Model answered 'no' but correct answer was 'center; upper-center'


### Example 2:
**Question:** How many polyps are in the image?
**Ground Truth:** 0
**Model Says:** There is one polyp in the image.
**Extracted Answer:** 1 ✗
**Problem:** Model answered '1' but correct answer was '0'


### Example 3:
**Question:** How many findings are present?
**Ground Truth:** 1
**Model Says:** There are two findings present in the image:

1. **Polyps**: The image shows several polyps, which are abnormal growths on the inner lining of the dig...
**Extracted Answer:** no ✗
**Problem:** Model answered 'no' but correct answer was '1'


### Example 4:
**Question:** Are there any abnormalities in the image? Check all that are present.
**Ground Truth:** polyp
**Model Says:** To determine if there are any abnormalities in the image, we need to carefully examine the details:

1. **Color and Texture**: The tissue appears to b...
**Extracted Answer:** no ✗
**Problem:** Model answered 'no' but correct answer was 'polyp'


### Example 5:
**Question:** Is this finding easy to detect?
**Ground Truth:** yes
**Model Says:** The image appears to show a medical procedure, likely an endoscopy, which is used to visualize the inside of the digestive tract. The presence of a la...
**Extracted Answer:** no ✗
**Problem:** Model answered 'no' but correct answer was 'yes'


================================================================================

## ❌ STAGE 2 FAILURE EXAMPLES (More Complex Challenges)


### Example 1:
**Question:** What type of polyp is present?
**Ground Truth:** none
**Model Says:** The image shows a polyp that appears to be a sessile polyp. Sessile polyps are flat and broad-based, often resembling a mushroom. They are typically f...
**Extracted Answer:** polyp ✗
**Problem:** Model answered 'polyp' but correct answer was 'none'


### Example 2:
**Question:** What is the size of the polyp?
**Ground Truth:** none
**Model Says:** The size of the polyp is 1 cm.
**Extracted Answer:** polyp ✗
**Problem:** Model answered 'polyp' but correct answer was 'none'


### Example 3:
**Question:** What color is the abnormality? If more than one separate with ;
**Ground Truth:** pink; red; white
**Model Says:** The abnormality appears to be red in color.
**Extracted Answer:** no ✗
**Problem:** Model answered 'no' but correct answer was 'pink; red; white'


### Example 4:
**Question:** Are there any abnormalities in the image? Check all that are present.
**Ground Truth:** polyp
**Model Says:** To determine if there are any abnormalities in the image, we need to carefully examine the tissue and any visible structures. Here are the steps to fo...
**Extracted Answer:** no ✗
**Problem:** Model answered 'no' but correct answer was 'polyp'


### Example 5:
**Question:** Are there any instruments in the image? Check all that are present.
**Ground Truth:** tube
**Model Says:** Yes, there is an instrument present in the image.
**Extracted Answer:** yes ✗
**Problem:** Model answered 'yes' but correct answer was 'tube'
