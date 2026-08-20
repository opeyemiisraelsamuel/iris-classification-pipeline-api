# Monitoring Model Performance Degradation in Production

## Why monitoring matters
A model's accuracy at deployment time is not permanent. Once live, the model
is exposed to real-world data that can drift away from what it was trained
on, and the relationships it learned can stop holding. Without monitoring,
degradation is silent — the API keeps returning predictions with full
confidence even as those predictions get worse.

## What can go wrong in production

**1. Data drift**
The distribution of incoming features starts to differ from the training
distribution. For example, if this model were retrained on a specific
population of users and the user base later shifts (new demographics, new
region, seasonality), the input values it sees will no longer resemble what
it learned from.

**2. Concept drift**
The relationship between inputs and the correct output changes over time,
even if the inputs look the same. The patterns that used to predict the
outcome correctly no longer do, because the real-world process generating
the data has changed.

**3. Label/feedback delay or absence**
In many production systems, the true outcome (ground truth) isn't known
immediately, or ever. This makes it hard to compute live accuracy directly,
so monitoring often has to rely on proxy signals instead.

## How I would monitor this model

**Log every prediction.** The API already logs each request's input
features, predicted class, and confidence score with a timestamp
(`prediction_log.jsonl`). This log is the raw material for everything below.

**Track data drift against a baseline.** At training time, the pipeline
saves the mean and standard deviation of each input feature, plus the
class distribution (`baseline_stats.json`). Periodically (e.g. daily or
weekly), I would compute the same statistics over a recent window of live
traffic and compare them to the baseline — for example using a
statistical test like the Kolmogorov-Smirnov test per feature, or simpler
threshold checks (e.g. "flag if a feature's mean shifts more than 2 standard
deviations from its training mean"). A significant shift signals the
incoming data no longer looks like what the model was trained on.

**Track confidence trends.** A drop in the model's average prediction
confidence, or a rise in the proportion of low-confidence predictions,
often signals the model is seeing inputs it's unsure about — a useful
early warning even before ground-truth labels are available.

**Track prediction distribution.** If the model suddenly starts predicting
one class far more or less often than the training class distribution
would suggest, that's another proxy signal worth flagging, independent of
whether the true labels are known yet.

**Compare against ground truth when available.** If true labels become
available later (e.g. through user feedback or manual review), I would
periodically recompute accuracy/F1 on that labeled sample and compare it to
the original evaluation numbers (`evaluation_report.txt`). A meaningful
drop is the clearest, most direct evidence of degradation.

**Set alert thresholds and a retraining trigger.** Rather than manually
watching dashboards, I'd define concrete thresholds (e.g. "alert if average
confidence drops more than 10% week-over-week" or "alert if any feature's
drift statistic crosses a set limit") and route them to an alert (email,
Slack). Sustained degradation would trigger a retraining pipeline using
more recent labeled data.

## Summary
Monitoring in production is less about a single metric and more about
layering signals: data drift (are the inputs still what we trained on?),
confidence/output trends (proxy signals available immediately), and
accuracy on labeled feedback when it exists (the ground truth check). Used
together, they catch degradation early enough to retrain before it
meaningfully affects users.
