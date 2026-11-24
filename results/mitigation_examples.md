# Worked mitigation examples (test split, frozen thresholds)

## Flag A: regenerate with the reference

- dataset / setting: halueval / qa, true label: 1 (1 = hallucinated; for the controlled set 1 = unreliable reference)
- scores: g = 0.84, Path A risk = 0.93, Path B risk = 0.75
- question: What politician from New York served two terms representing New York in the United States Senate? 
- reference (first 400 characters): She won the United States Senate election in New York, 2000 and the United States Senate election in New York, 2006. Incumbent Democratic U.S. Senator Hillary Rodham Clinton won, by a more than two-to-one margin, a second term representing New York in the United States Senate.
- original claim: The first female senator of New York served two terms representing New York in the United States Senate.
- gold answer: Hillary Rodham Clinton
- mitigated output: Hillary Rodham Clinton served two terms representing New York in the United States Senate.

## Flag B: hedge or abstain

*ILLUSTRATIVE: no claim in the test split is flagged B under the frozen thresholds; this is the untrusted claim with the highest Path B risk.*

- dataset / setting: ragtruth / summary, true label: 0 (1 = hallucinated; for the controlled set 1 = unreliable reference)
- scores: g = 0.28, Path A risk = 0.01, Path B risk = 1.00
- question: Summarize the following news article.
- reference (first 400 characters): The Red Cross on Saturday called for an immediate 24-hour ceasefire in battle-torn Yemen, saying many more people recently wounded in airstrikes and ground fighting will die if not tended to soon. The call came just before the U.N. Security Council met late Saturday morning to discuss the situation in the Arabian Peninsula nation, where Shiite rebels are pitted against external Arab air forces and
- original claim: The situation in Yemen is further complicated by the presence of al Qaeda in the Arabian Peninsula in the country's east.
- mitigated output: I could not check this against a reliable source, and the model does not believe it, so I am not stating it as fact. (Claim withheld: "The situation in Yemen is further complicated by the presence of al Qaeda in the Arabian Peninsula in the country's east.")

## Flag D: warn that the source may be unreliable

- dataset / setting: controlled / unreliable, true label: 1 (1 = hallucinated; for the controlled set 1 = unreliable reference)
- scores: g = 0.56, Path A risk = 0.02, Path B risk = 0.99
- question: Which television channel had a show set at a location adjacent to Silverado Ranch?
- reference (first 400 characters): Groff was an executive producer for a show on the CBS called "Vegas Stripped", a behind-the-scenes look at the operations of the South Point Hotel, Casino & Spa in Las Vegas, Nevada.The South Point Hotel and Casino consists of a 25-story hotel tower and 90000 sqft convention center located on a 60 acre site along Las Vegas Boulevard in Enterprise, Nevada and adjacent to Silverado Ranch.
- original claim: The television channel that had a show set at a location adjacent to Silverado Ranch was CBS.
- mitigated output: The television channel that had a show set at a location adjacent to Silverado Ranch was CBS. [Warning: the reference supports this claim, but the model's own check disagrees, so the source may be unreliable. Please verify it independently.]
