# Performance

## Calculating returns

If at the beginning of a time period $i$ an asset value is $V_{i-1}$ and
at the end of the period the asset value is $V_i$, then the `profit and loss`
for this period is
$$PnL_i=V_i-V_{i-1}$$
$PnL$ is in absolute unts (euros,dollars, cents),
which makes it difficult to compare `PnL`s of different assets.

The [`return`](https://en.wikipedia.org/wiki/Rate_of_return) over this period is
$$ R_i=\frac{V_i - V_{i-1}}{V_{i-1}}=\frac{V_i}{V_{i-1}}-1$$
We use returns to track the performance of a trading strategy over a certain period.
Here we implicitely assume $V_i$ and $V_{i-1}$ are measured with a constant time interval,
$t_i-t_{i-1}$.

If we have to compare returns over different periods
(for instance returns of trade rountdtips, every roundtrip having it's own time span),
we need to normalize them with respect to time, making returns evenly spaced in time.

The `rate of return` is
$$r_i=\frac{R_i}{t_i-t_{i-1}}$$
Obviously, here $r_i$ depends on the units of time measurement.

In finance, the time units are years, and the $t_i - t_{i-1}$
are expressed as `fractional years`.
This process is called `annualization`and $r_i$ is called `annualized return`.
Since an annualized rate of return over a period of less than one year is statistically unlikely to be indicative of returns over the long run,
the [CFA Institute's Global Investment Performance Standards (GIPS)](https://www.cfainstitute.org/en/membership/professional-development/refresher-readings/gips-overview) says
> Returns for periods of less than one year must not be annualized.

In intraday trading, we can use `fractional days` to represent the $t_i - t_{i-1}$.
I haven't seen any standard name for this (`dailyzation` sounds silly) normalization.

To measure the performance of an account, we calculate `annualized` or `daylylized`
returns of account balance at some regular time intervals (for instance at the every
step environment makes).

To measure the performance of a roundtrip, we calculate the [`return on investment`](https://en.wikipedia.org/wiki/Return_on_investment)
$$ roi = \frac{V_{end}}{V{start}+commission}-1$$

## Risk-adjested return ratios

The following is heavily influenced by the
["Measures of Risk-adjusted Return"](https://www.turingfinance.com/computational-investing-with-python-week-one/)
article by [Stuart Gordon Reid](https://github.com/StuartGordonReid/).

Shortened version is [here](https://github.com/laholmes/risk-adjusted-return/blob/master/app.py)

What is the risk in financial trading?
This is a philosophical question, but generally risk in any investment is the probability of loss.
For most of finance this means risk-adjusted return.

Some terminology:

- Risk-adjusted returns measure how many units of excess return are expected to be generated from however many units of risk.
- Excess return is the return of the investment above either a benchmark, risk-free rate of return, or some minimum required rate of return.

There are several popular risk-adjusted return measures:

- Volatility of historical returns. Volatility assumes that the riskiness of a security is how much much it moves around i.e. it's volatility. The most common volatility based measure of risk is the standard deviation of historical returns.
- Lower partial moments. Lower partial moments argue that risk is only captured in the downside of the historical volatility of the portfolio. An example of a lower partial moment would be the standard deviation of negative returns.
- Drawdown risk is the maximum historical 'drawdown' of the portfolio. A drawdown is the percentage loss between peak and trough.
- Expected shortfalls. Expected shortfall argues that the risk of a portfolio is the dollar value which could reasonably be expected to be lost over a specified period of time given a pre-specified confidence interval. The most popular measure of expected shortfall risk is Value at Risk (VaR).

When we "discount" expected return by different quantities of risk we get measures of risk-adjusted return.

| Descriptive statistics | Absolute | With benchmark | Partial moments: downside | Partial moments: gain-loss | Drawdown |
| --- | --- | --- | --- | --- | --- |
| 1st moment, mean | MAD ratio | Relative batting average | Omega-Sharpe ratio | Omega ratio | Sterling ratio |
| 2nd moment, variability | Sharpe ratio | Information ratio | Sortino ratio | Variability skewness | Burke ratio |
| 3rd moment, skewness | Skew-adjusted Sharpe ratio | Skew-adjusted information ratio | Kappa 3 ratio | Gain-loss skewness | Calmar ratio |
| 4th moment, kurtosis | Adjusted Sharpe ratio | Adjusted information ratio | Sortino-Satchell ratio | Farnelli - Tibiletti ratio | Pain ratio |
| systematic risk, beta | Treynor ratio | Appraisal ratio | Return to duration | Timing ratio | Ulcer ratio |
| Extreme risk, VaR | Reward to VaR | Reward to relative VaR | Conditional Sharpe ratio | Rachev ratio | Reward to conditional drawdown |
| Other | K ratio | Upside capture | Downside capture | Capture ratio | Bias ratio |
| Other | Absolute batting average | Risk efficiency ratio | Tail ratio | Convexity | Active share |

### Descriptive statistics

#### Skewness

[src](https://github.com/braverock/PerformanceAnalytics/blob/master/R/skewness.R)

Compute skewness of a univariate distribution.

This function was ported from the RMetrics package fUtilities to eliminate a
dependency on fUtiltiies being loaded every time.
$$Skewness(moment) = \frac{1}{n}*\sum^{n}_{i=1}(\frac{r_i - \overline{r}}{\sigma_P})^3$$

$$Skewness(sample) =  \frac{n}{(n-1)*(n-2)}*\sum^{n}_{i=1}(\frac{r_i - \overline{r}}{\sigma_{S_P}})^3$$

$Skewness(fisher) = \frac{\frac{\sqrt{n*(n-1)}}{n-2}*\sum^{n}_{i=1}\frac{x^3}{n}}{\sum^{n}_{i=1}(\frac{x^2}{n})^{3/2}}$$

where $n$ is the number of return, $\overline{r}$ is the mean of the return
distribution, $\sigma_P$ is its standard deviation and $\sigma_{S_P}$ is its
sample standard deviation

- param method a character string which specifies the method of computation.
  These are either "moment" or "fisher".
  
  The "moment" method is based on the definitions of skewnessfor distributions;
  these forms should be used when resampling (bootstrap or jackknife).
  
  The "fisher" method correspond to the usual "unbiased" definition of sample
  variance, although in the case of skewness exact unbiasedness is not possible.
  The "sample" method gives the sample skewness of the distribution.

We use `scipy.stats` package to compute the sample skewness as the
Fisher-Pearson coefficient of skewness.
When the `bias` parameter (`True` or `False`), unbiased skewness
$$g_1=\frac{m_3}{m_2^{3/2}}$$
and biased scewness
$$G_1=\frac{\sqrt{N(N-1)}}{N-2}\frac{m_3}{m_2^{3/2}}$$
where the $i^{th}$ biased sample moment
$$m_i=\frac{1}{N}\sum_{n=1}^N(x[n]-\bar{x})^i$$
and $\bar{x}$ is the sample mean.

#### Kurtosis

We use `scipy.stats` package to compute the sample skewness as the
kurtosis.

The `fisher` parameter (`True` or `False`) defines if
Fisher's (normal ==> 0.0) or Pearson's (normal ==> 3.0) definition is used.

If the `bias` parameter is `False`, then the calculations
are corrected for statistical bias.

### Measures of risk-adjusted return based on volatility

For a given period of time standard deviation, , measures the historical variance (average of the squared deviations) of the returns from the mean return, , over that period of time. The formula for this is,

where

Beta measures the relationship between the security returns,  and the market, . High beta stocks are considered to be more risk whereas low beta stocks are considered to be less risky. The formula for this is

#### Treynor ratio

[wikipedia](https://en.wikipedia.org/wiki/Treynor_ratio)

The Treynor ratio was one of the first measures of risk-adjusted return. It was originally published in 1965 in the Harvard Business Review as a metric for rating the performance of investment funds. Given a risk-free rate of return, , the Treynor ratio calculates the excess returns generated by a portfolio, , and discounts it by the portfolio's beta, ,

#### Sharpe ratio

[wikipedia](https://en.wikipedia.org/wiki/Sharpe_ratio)

The Sharpe ratio, originally called the reward-to-variability ratio,
was introduced in [1966 by William Sharpe](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm) as an extension of the
Treynor ratio.

The Sharpe Ratio divides the return of a portfolio in excess of the risk
free rate by its standard deviation (volatility of the returns).

#### Information ratio

[wikipedia](https://en.wikipedia.org/wiki/Information_ratio)

The information ratio is an extension of the Sharpe ratio which replaces the risk-free rate of return with the scalar expected return of a benchmark portfolio, ,

#### Modigliani ratio

[wikipedia](https://en.wikipedia.org/wiki/Modigliani_risk-adjusted_performance)

The Modigliani ratio a.k.a the M2 ratio, is a combination the Sharpe and information ratio in that it adjusts the expected excess returns of the portfolio above the risk free rate by the expected excess returns of a benchmark portfolio, , or the market , above the risk free rate,

### Measures of risk-adjusted return based on lower partial moments

Whereas measures of risk-adjusted return based on volatility treat all deviations from the mean as risk, measures of risk-adjusted return based on lower partial moments consider only deviations below some predefined minimum return threshold,  as risk. For example, negative deviations from the mean is risky whereas positive deviations are not. A lower partial moment of order  can be estimated from a sample of  returns as follows,

where is historical returns.

A useful classification of measures of risk-adjusted returns based on lower partial moments in by their order. The larger the order the greater the weighting will be on returns that fall below the target threshold, meaning that larger orders result in more risk-averse measures. In addition to lower partial moment which can be a measure of downside risk in a portfolio, higher partial moments can be measures of the upside potential of a portfolio,

In some ways, Value at Risk (VaR) is similar to a lower partial moment, except that VaR is of order 2 only and is a more probabilistic view of loss as the risk of a portfolio. Personally I prefer lower partial moments as risk.

#### Omega ratio

The [Ω ratio](https://en.wikipedia.org/wiki/Omega_ratio) is a probability weighted ratio of gains versus losses
for some `target return` threshold. This `target return` separates gains
from losses.

It was proposed by [Keating and Shadwick in 2002](https://web.archive.org/web/20190804141428/https://pdfs.semanticscholar.org/a63b/0a002c6cf2d4085f7ad80cbfd92fe3520521.pdf),
referred to as `Gamma` in their original paper.

Mathematically, the `Ω` ratio is:
$$\Omega(TR)={\frac {\int _{TR}^{\infty }[1-F(r)]\,dr}{\int _{-\infty }^{TR}F(r)\,dr}}$$
where $F(r)$ is the cumulative probability distribution function
of the returns $r$ and $TR$ is the `target return` threshold.

The equation ablve seems very difficult to calculate, but,
fortunatelly, in 2003
[Kazemi, Schneeweis, and Gupta in "Omega as a Performance Measure"](https://people.duke.edu/~charvey/Teaching/BA453_2006/Schneeweis_Omega_as_a.pdf)
showed that Omega can be written as:
$$\Omega(TR)=\frac {E[\max(x - TR, 0)]}{E[max(TR - x, 0)]}$$
which is a ratio of the high partial moment of order 1
to the lower partial moment of order 1 and can be easily calculated.

The equation ablve seems very difficult to calculate, but,
fortunatelly, in 2004
[P. D. Kaplan and J. A. Knowles in "Kappa: A generalized downside risk-adjusted performance measure"](http://w.performance-measurement.org/KaplanKnowles2004.pdf)
showed that Omega can be written as:
$$\Omega(TR)=\frac{R_{p}-TR}{LPM_1}+1$$
which can be easily calculated.

References:

Keating, J. and Shadwick, W.F. **A Universal Performance Measure**
The Finance Development Centre Limited, London. 2002.
[Archived PDF](https://web.archive.org/web/20190804141428/https://pdfs.semanticscholar.org/a63b/0a002c6cf2d4085f7ad80cbfd92fe3520521.pdf)
from the
[original PDF](https://people.duke.edu/~charvey/Teaching/BA453_2006/Keating_A_universal_performance.pdf)
on 2019-08-04.
[S2CID 16222368](https://api.semanticscholar.org/CorpusID:16222368).

Kazemi, Schneeweis, and Gupta. **Omega as a Performance Measure**. 2003.
[original PDF](https://people.duke.edu/~charvey/Teaching/BA453_2006/Schneeweis_Omega_as_a.pdf)

Paul D. Kaplan and James A. Knowles,
**Kappa: A generalized downside risk-adjusted performance measure**,
Miscellaneous Publication, Morningstar Associates and York Hedge Fund Strategies,
2004
[PDF offline](./KaplanKnowles2004.pdf)
[PDF online](http://w.performance-measurement.org/KaplanKnowles2004.pdf)
[PDF at research gate](https://www.researchgate.net/publication/284690156_Kappa_A_Generalized_Downside_Risk-Adjusted_Performance_Measure)

#### Sortino ratio

The [Sortino ratio](https://en.wikipedia.org/wiki/Sortino_ratio) was proposed as a modification to the Sharpe ratio by Sortino and van der Meer in 1991. The Sortino ratio discounts the excess return of a portfolio above a target threshold by the volatility of downside returns, , instead of the volatility of all returns, . The volatility of downside returns is equivalent to the square-root second-order lower partial moment of returns,

#### Kappa ratio

[wikipedia](https://en.wikipedia.org/wiki/???)
[R doc](https://www.rdocumentation.org/packages/PerformanceAnalytics/versions/1.1.0/topics/Kappa)
[doc](https://breakingdownfinance.com/finance-topics/performance-measurement/kappa-ratio/)

The Kappa ratio is a generalization of Omega and Sortino ratios first proposed in 2004 by Kaplan and Knowles.  It was shown that when the parameter  of the Kappa ratio is set to one or two you get the Omega or Sortino ratio. The Kappa ratio is most often used with  which is why it is often referred to as the Kappa 3 ratio,

[src](https://github.com/braverock/PerformanceAnalytics/blob/master/R/Kappa.R)

Introduced by Kaplan and Knowles (2004), Kappa is a generalized
downside risk-adjusted performance measure.

To calculate it, we take the difference of the mean of the distribution
to the target and we divide it by the l-root of the lth lower partial
moment. To calculate the lth lower partial moment we take the subset of
returns below the target and we sum the differences of the target to
these returns. We then return return this sum divided by the length of
the whole distribution.

$$Kappa(MAR, order) = \frac{R_{p}-MAR}{\sqrt[order]{\frac{1}{n}*\sum^n_{t=1}
max(MAR-R_{t}, 0)^{order}}}$$
$$= \frac{R_{p}-MAR}{\sqrt[order]{LPM_{order}}}$$

For l=1 kappa is the Sharpe-omega ratio and for l=2 kappa
is the sortino ratio.

Kappa should only be used to rank portfolios as it is difficult to
interpret the absolute differences between kappas. The higher the
kappa is, the better.

References

Paul D. Kaplan and James A. Knowles,
**Kappa: A generalized downside risk-adjusted performance measure**,
Miscellaneous Publication, Morningstar Associates and York Hedge Fund Strategies,
2004
[PDF offline](./KaplanKnowles2004.pdf)
[PDF online](http://w.performance-measurement.org/KaplanKnowles2004.pdf)
[PDF at research gate](https://www.researchgate.net/publication/284690156_Kappa_A_Generalized_Downside_Risk-Adjusted_Performance_Measure)

#### Bernardo Ledoit (gain-loss) ratio

[wikipedia](https://en.wikipedia.org/wiki/???)

The gain-loss ratio was first presented by Bernardo Ledoit in 2000. It discounts the first-order higher partial moment of a portfolio's returns, upside potential, by the first-order lower partial moment of a portfolio's returns, downside risk,

It is a special case of the omega ratio when $TR$ (`target return`) is set to zero.

Bernardo, A. and Ledoit, O. (2000) Gain, Loss and Asset Pricing. Journal of the Political Economy, 108, 144-172.
https://doi.org/10.1086/262114

src](https://github.com/braverock/PerformanceAnalytics/blob/master/R/BernadoLedoitratio.R)

To calculate Bernardo and Ledoit ratio we take the sum of the subset of
returns that are above 0 and we divide it by the opposite of the sum of
the subset of returns that are below 0
$$BernardoLedoitRatio(R) = \frac{\frac{1}{n}\sum^{n}_{t=1}{max(R_{t},0)}}{\frac{1}{n}\sum^{n}_{t=1}{max(-R_{t},0)}}$$
where $n$ is the number of observations of the entire series

References

Carl Bacon, **Practical portfolio performance measurement and attribution**,
second edition 2008 Wiley p.95
ISBN:9780470059289 |Online ISBN:9781119206309 |DOI:10.1002/9781119206309

Bernardo, Antonio E.; Ledoit, Olivier (2000-02-01). **Gain, Loss, and Asset Pricing**.
Journal of Political Economy. 108 (1): 144–172.
[CiteSeerX 10.1.1.39.2638](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.39.2638)
[doi:10.1086/262114](https://doi.org/10.1086%2F262114)
[ISSN 0022-3808](https://www.worldcat.org/issn/0022-3808)
[S2CID 16854983](https://api.semanticscholar.org/CorpusID:16854983).

#### Upside-potential ratio

[wikipedia](https://en.wikipedia.org/wiki/Upside_potential_ratio)

The upside-potential ratio was first presented by Sortino van  der Meer and Plantinga in 1999. It discounts the first-order higher partial moment of a portfolio's returns, upside potential, by the square root of the second-order lower partial moment of a portfolio's returns, downside variation,

[src](https://github.com/braverock/PerformanceAnalytics/blob/master/R/UpsidePotentialRatio.R)

Calculates Upside Potential Ratio of upside performance over downside risk

Sortino proposed an improvement on the Sharpe Ratio to better account for
skill and excess performance by using only downside semivariance as the
measure of risk.  That measure is the `SortinoRatio`. This
function, Upside Potential Ratio, was a further improvement, extending the
measurement of only upside on the numerator, and only downside of the
denominator of the ratio equation.

Sortino contends that risk should be measured in terms of not meeting the
investment goal.  This gives rise to the notion of `Minimum Acceptable Return`
or `MAR`.  All of Sortino's proposed measures include the
`MAR`, and are more sensitive to downside or extreme risks than measures that
use volatility(standard deviation of returns) as the measure of risk.

Choosing the MAR carefully is very important, especially when comparing
disparate investment choices.  If the MAR is too low, it will not adequately
capture the risks that concern the investor, and if the MAR is too high, it
will unfavorably portray what may otherwise be a sound investment.  When
comparing multiple investments, some papers recommend using the risk free
rate as the MAR.  Practitioners may wish to choose one MAR for consistency,
several standardized MAR values for reporting a range of scenarios, or a MAR
customized to the objective of the investor.
$$UPR=\frac{\sum^{n}_{t=1}(R_{t} - MAR)}{\delta_{MAR}}$$
where $delta_{MAR}$ is the `DownsideDeviation`.

The numerator in `UpsidePotentialRatio` only uses returns that exceed
the MAR, and the denominator (in `DownsideDeviation`) only uses
returns that fall short of the MAR by default.  Sortino contends that this
is a more accurate and balanced protrayal of return potential, wherase
`SortinoRatio` can reward managers most at the peak of a cycle,
without adequately penalizing them for past mediocre performance.  Others
have used the full series, and this is provided as an option by the
`method` argument.

- param MAR Minimum Acceptable Return, in the same periodicity as your returns
- param method one of "full" or "subset", indicating whether to use the
  length of the full series or the length of the subset of the series
  above(below) the MAR as the denominator, defaults to "subset"

Plantinga, A., van der Meer, R. and Sortino, F.
**The Impact of Downside Risk on Risk-Adjusted Performance of Mutual Funds in the Euronext Markets**
July 19, 2001. [SSRN](http://ssrn.com/abstract=277352) [local PDF](./ssrn_id277352_code010721600.pdf)

### Measures of risk-adjusted return based on drawdowns

A [drawdown](https://en.wikipedia.org/wiki/Drawdown_(economics))
is the decrease from a historical peak in the cumulative returns.
Drawdown might be defined as either

- the peak-to-valley fall in performance or
- any continuous, uninterrupted losing return period.

Given a series of cumulative returns $R_i$, $i=0,\ldots ,n$
$$DD_n=\max _{i\in (0,n)}R_i-R_n$$
where $DD_n$ is the drawdown at the period $n$.

The `average drawdown` is the mean of a sequence of drawdowns.

The `maximum drawdown` is the absolute value of the minimum in a sequence of drawdowns.

#### Calmar ratio

[wikipedia](https://en.wikipedia.org/wiki/Calmar_ratio)

The Calmar ratio discounts the expected excess return of a portfolio by the worst expected maximum draw down for that portfolio,

#### Sterling ratio

[wikipedia](https://en.wikipedia.org/wiki/Sterling_ratio)
[doc](https://breakingdownfinance.com/finance-topics/performance-measurement/sterling-ratio/)

The Sterling ratio discounts the expected excess return of a portfolio by the average of the  worst expected maximum drawdowns for that portfolio,

#### Burke ratio

[wikipedia](https://en.wikipedia.org/wiki/???)

The Burke ratio is similar to the Sterling ratio except that it is less sensitive to outliers. It discounts the expected excess return of a portfolio by the square root of the average of the  worst expected maximum drawdowns squared for that portfolio,

[src](https://github.com/braverock/PerformanceAnalytics/blob/master/R/BurkeRatio.R)

Burke ratio of the return distribution

[doc](https://breakingdownfinance.com/finance-topics/performance-measurement/burke-ratio/)

To calculate Burke ratio we take the difference between the portfolio
return and the risk free rate and we divide it by the square root of the
sum of the square of the drawdowns. To calculate the modified Burke ratio
we just multiply the Burke ratio by the square root of the number of datas.
$$Burke Ratio = \frac{r_P - r_F}{\sqrt{\sum^{d}_{t=1}{D_t}^2}}$$
$$Modified Burke Ratio = \frac{r_P - r_F}{\sqrt{\sum^{d}_{t=1}\frac{{D_t}^2}{n}}}$$
where $n$ is the number of observations of the entire series, $d$ is number of drawdowns,
$r_P$ is the portfolio return, $r_F$ is the risk free rate and $D_t$ the $t^{th}$ drawdown.

- param Rf the risk free rate
- param modified a boolean to decide which ratio to calculate between Burke ratio and modified Burke ratio.

references

Carl Bacon, **Practical portfolio performance measurement and attribution**, second edition 2008 p.90-91

```R
BurkeRatio <- function (R, Rf = 0, modified = FALSE, ...)
{
    drawdown = c()
    n = length(R)

    number_drawdown = 0
    in_drawdown = FALSE
    peak = 1

    period = Frequency(R) # we want this to be 1
    for (i in (2:length(R))) {
        if (R[i]<0)
        {
            if (!in_drawdown)
            {
                peak = i-1
                number_drawdown = number_drawdown + 1
                in_drawdown = TRUE
            }
        }
        else
        {
            if (in_drawdown)
            {
                temp = 1
                boundary1 = peak+1
                boundary2 = i-1
                for(j in (boundary1:boundary2)) {
                    temp = temp*(1+R[j]*0.01)
                }
                drawdown = c(drawdown, (temp - 1) * 100)
                in_drawdown = FALSE
            }
        }
    }
    if (in_drawdown)
    {
        temp = 1
        boundary1 = peak+1
        boundary2 = i
        for(j in (boundary1:boundary2)) {
            temp = temp*(1+R[j]*0.01)
        }
        drawdown = c(drawdown, (temp - 1) * 100)
        in_drawdown = FALSE
    }

    D = Drawdowns(R) # Somehow they don't use it

    Rp = (prod(1 + R))^(period / length(R)) - 1
    result = (Rp - Rf)/sqrt(sum(drawdown^2))
    if(modified) {
        result = result * sqrt(n)
    }
    return(result)
}
```

#### Pain ratio

[src pain index](https://github.com/braverock/PerformanceAnalytics/blob/master/R/PainIndex.R)
[src pain ratio](https://github.com/braverock/PerformanceAnalytics/blob/master/R/PainRatio.R)

Pain index of the return distribution

The pain index is the mean value of the drawdowns over the entire
analysis period. The measure is similar to the Ulcer index except that
the drawdowns are not squared.  Also, it's different than the average
drawdown, in that the numerator is the total number of observations
rather than the number of drawdowns.

Visually, the pain index is the area of the region that is enclosed by
the horizontal line at zero percent and the drawdown line in the
Drawdown chart.
$$Pain index = \sum^{n}_{i=1} \frac{\mid D'_i \mid}{n}$$
where $n$ is the number of observations of the entire series, $D'_i$ is
the drawdown since previous peak in period $i$.

```R
PainIndex <- function (R, ...) {
    result = sum(abs(DrawdownPeak(R)))
    R = na.omit(R)
    result = result / length(R)
    return(result)
}
```

Pain ratio of the return distribution

To calculate Pain ratio we divide the difference of the portfolio return
and the risk free rate by the Pain index
$$Pain ratio = \frac{r_P - r_F}{\sum^{n}_{i=1} \frac{\mid D'_i \mid}{n}}$$
where $r_P$ is the annualized portfolio return, $r_F$ is the risk free
rate, $n$ is the number of observations of the entire series, $D'_i$ is
the drawdown since previous peak in period $i$

- param Rf risk free rate, in same period as your returns

```R
PainRatio <- function (R, Rf = 0, ...){
    PI = PainIndex(R)
    n = length(R)
    period = Frequency(R)
    Rp = (prod(1 + R))^(period / length(R)) - 1
    result = (Rp - Rf) / PI
    return(result)
}
```

#### Martin ratio (Ulcer ratio)

[src ulcer index](https://github.com/braverock/PerformanceAnalytics/blob/master/R/UlcerIndex.R)
[src ulcer ratio](https://github.com/braverock/PerformanceAnalytics/blob/master/R/MartinRatio.R)

The Ulcer Index

Developed by Peter G. Martin in 1987 (Martin and McCann, 1987) and named
for the worry caused to the portfolio manager or investor.  This is
similar to drawdown deviation except that the impact of the duration of
drawdowns is incorporated by selecting the negative return for each
period below the previous peak or high water mark.  The impact of long,
deep drawdowns will have significant impact because the underperformance
since the last peak is squared.
$$UI = \sqrt{\sum^{n}_{i=1}{\frac{{D'_i}^2}{n}}}$$
where $D'_i$ is the drawdown since previous peak in period $i$.

Reference

Martin, P. and McCann, B. (1989)
**The investor's Guide to Fidelity Funds: Winning Strategies for Mutual Fund Investors**
John Wiley & Sons, Inc.
This out-of-print book is [available in PDF](http://www.tangotools.com/ui/fkbook.pdf.)
on [Peter Martin's web page](http://www.tangotools.com/ui/ui.htm)
On his page there is also an [Excel spreadsheet](http://www.tangotools.com/ui/UlcerIndex.xls)
with original calculations of the Ulcer index.

```R
UlcerIndex <- function (R, ...) {
    result = sqrt(sum(DrawdownPeak(R)^2))
    return (result/sqrt(length(R)))
}
```

Martin ratio of the return distribution

Also called Ulcer ratio, Ulcer Performance Index (UPI)

To calculate Martin ratio we divide the difference of the portfolio return
and the risk free rate by the Ulcer index
$$Martin ratio = \frac{r_P - r_F}{\sqrt{\sum^{n}_{i=1} \frac{{D'_i}^2}{n}}}$$
where $r_P$ is the annualized portfolio return, $r_F$ is the risk free
rate, $n$ is the number of observations of the entire series, $D'_i$ is
the drawdown since previous peak in period $i$.

- param Rf risk free rate, in same period as your returns

```R
MartinRatio <- function (R, Rf = 0, ...) 
{
    period = Frequency(R)
    UI = UlcerIndex(R)
    n = length(R)
    Rp = (prod(1 + R))^(period / length(R)) - 1
    return((Rp - Rf) / UI)
}
```

#### Risk–return ratio

[wikipedia](https://en.wikipedia.org/wiki/Risk%E2%80%93return_ratio)

### Measures of risk-adjusted return based on Expected Shortfall (Value at Risk)

Value at Risk (VaR) is the most popular measure of expected shortfall. Expected shortfall works as follows: given a specific time period, , and confidence interval, , expected shortfall tells us what the maximum probable loss scenario is over that period of time (usually one day a.k.a. 1-day VaR) with a probability of . There are three approaches to calculating VaR, historical simulation VaR, delta-normal VaR, and Monte Carlo VaR.

Historical simulation VaR takes historical  period returns, orders then, and takes the loss at the point in the list which corresponds to . For example, if , , and we have the following 10 returns: , then the item in the list which corresponds to  is -4.5%. This can be interpreted as us either being 90% sure that -4.5% is our expected 1-day shortfall for the portfolio or, alternatively, that 90% of the time a 1-day loss experienced by the portfolio won't exceed -4.5%.

Delta-normal VaR assumes that the returns generated by the assets in the portfolio follow a pre-specified distribution. Unfortunately a popular assumption is that returns are normally distributed despite the fact that in reality portfolio returns exhibit fatter tails meaning that the probability of outliers (significant gains and losses) is higher. Given these assumptions it is possible to calculate what the returns and standard deviation of the portfolio should be as a whole. From this the  worst case scenario for the portfolio can be estimated.

Monte Carlo VaR works by simulating the portfolio using stochastic processes. This can be done in two ways. Either a stochastic process is calibrated to each asset in the portfolio, return paths for each asset are simulated, and then these paths are combined given some correlation matrix using the Cholesky Decomposition; or a stochastic process is calibrated to the historical returns of the portfolio and return paths for the portfolio are simulated. Depending on your application one method may be considerably more or less computational expensive. Once returns have been simulated for the portfolio the "historical simulation" VaR method is applied to the returns.

There are many problems with Value at Risk. One problem with VaR is that it violates the sub-additive rule of risk which requires that the risk of a portfolio cannot exceed the risk of it's constituent assets i.e. diversification cannot be negative. For this reason variants of VaR such as Conditional VaR and Extreme VaR have been proposed.

#### Excess return on VaR

The excess return on Value at Risk discounts the excess return of the portfolio above the risk-free rate by the Value at Risk of the portfolio,

#### Conditional Sharpe ratio

The "conditional Sharpe ratio" discounts the excess return of the portfolio above the risk-free rate by the Conditional Value at Risk of the portfolio,

## Uneven returns in time

We're taking care of the uneven spacing by normalizing the returns with respect to time.
$$v=\sum_i (\frac{x_{i+1}-x_i}{t_{i+1}-t_i})^2$$
$$\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} \frac{\log(S_i / S_{i-1})^2}{t_i - t_{i-1}}$$

Some links:
[1](https://quant.stackexchange.com/questions/42407/estimating-daily-volatility-of-unevenly-irregularly-spaced-time-series-data),
[2](https://quant.stackexchange.com/questions/2616/how-do-you-estimate-the-volatility-of-a-sample-when-points-are-irregularly-space),
[3](https://quant.stackexchange.com/questions/2565/how-to-interpolate-gaps-in-a-time-series-using-closely-related-time-series),
[pdf](http://eckner.com/papers/unevenly_spaced_time_series_analysis.pdf)

Cumulative geometric (compounded) return is a product of all the individual period returns
$$(1+r_{1})(1+r_{2})(1+r_{3})\ldots(1+r_{n})-1=prod(1+R)-1$$

## Sharpe calculation

The Smart Sharpe Ratio formula includes an Autocorrelation Penalty term (ρ_i) in the denominator:
Smart Sharpe Ratio = (Rp - Rf) / √(σ^2 + ρ_i * σ^2)
where:
Rp is the portfolio return
Rf is the risk-free rate
σ is the portfolio standard deviation
ρ_i are the correlation coefficients of the time series and its lags
[code](https://forum.numer.ai/t/performance-stationarity/151/6)
[pdf](https://www.keyquant.com/Download/GetFile?Filename=%5CPublications%5CKeyQuant_WhitePaper_APT_Part2.pdf)
[pdf 2](https://cran.r-project.org/web/packages/SharpeR/vignettes/SharpeRatio.pdf)
[pdf 3](https://www.twosigma.com/wp-content/uploads/sharpe-tr-1.pdf)

[Calculation is based on this paper by Red Rock Capital](http://www.redrockcapital.com/Sortino__A__Sharper__Ratio_Red_Rock_Capital.pdf)

[general tra strategy](https://github.com/jingmouren/gitee-yunjinqi-backtrader/tree/4746afeca86b363681871e0471074bdee0991db1)

### R Skewness-Kurtosis ratio

Skewness-Kurtosis ratio of the return distribution
Skewness-Kurtosis ratio is the division of Skewness by Kurtosis.
It is used in conjunction with the Sharpe ratio to rank portfolios.
The higher the rate the better.
$$SkewnessKurtosisRatio(R , MAR) = \frac{S}{K}$$
where $S$ is the skewness and $K$ is the Kurtosis

```R
SkewnessKurtosisRatio <-
function (R, ...)
{
    R = checkData(R)

    if (ncol(R)==1 || is.null(R) || is.vector(R)) {
       calcul = FALSE
        for (i in (1:length(R))) {
            if (!is.na(R[i])) {
               calcul = TRUE
            }
        }
        R = na.omit(R)
        if(!calcul) {
            result = NA
        }
        else {
            result = skewness(R, method = "moment") / kurtosis(R, method = "moment")
        }
        return(result)
    } else {
        result = apply(R, MARGIN = 2, SkewnessKurtosisRatio, ...)
        result<-t(result)
        colnames(result) = colnames(R)
        rownames(result) = paste("SkewnessKurtosisRatio", sep="")
        return(result)
    }
}
```

### R Sortino Ratio

[src](https://github.com/braverock/PerformanceAnalytics/blob/master/R/SortinoRatio.R)

calculate Sortino Ratio of performance over downside risk

Sortino proposed an improvement on the Sharpe Ratio to better account for
skill and excess performance by using only downside semivariance as the
measure of risk.

Sortino contends that risk should be measured in terms of not meeting the
investment goal.  This gives rise to the notion of `Minimum
Acceptable Return` or `MAR`.  All of Sortino's proposed measures include the
`MAR`, and are more sensitive to downside or extreme risks than measures that
use volatility(standard deviation of returns) as the measure of risk.

Choosing the MAR carefully is very important, especially when comparing
disparate investment choices.  If the MAR is too low, it will not adequately
capture the risks that concern the investor, and if the MAR is too high, it
will unfavorably portray what may otherwise be a sound investment.  When
comparing multiple investments, some papers recommend using the risk free
rate as the MAR.  Practitioners may wish to choose one MAR for consistency,
several standardized MAR values for reporting a range of scenarios, or a MAR
customized to the objective of the investor.
$$SortinoRatio=\frac{(\overline{R_{a} - MAR})}{\delta_{MAR}}$$
where
$\delta_{MAR}$ is the `DownsideDeviation`.

- param MAR Minimum Acceptable Return, in the same periodicity as your returns

references

Sortino, F. and Price, L. Performance Measurement in a Downside Risk Framework. **Journal of Investing**. Fall 1994, 59-65.

```R
SortinoRatio <-
function (R, MAR = 0,...,
          weights=NULL,
          SE=FALSE, SE.control=NULL)
{ # @author Brian G. Peterson
  # modified from function by Sankalp Upadhyay <sankalp.upadhyay [at] gmail [dot] com> with permission

    # Description:
    # Sortino proposed to better account for skill and excess peRformance
    # by using only downside semivariance as the measure of risk.

    # R     return vector
    # MAR   minimum acceptable return
    # Function:
    R = checkData(R)
    
    #if we have a weights vector, use it
    if(!is.null(weights)){
        R=Return.portfolio(R,weights,...)
    }
    
    sr <-function (R, MAR)
    {
        SR = mean(Return.excess(R, MAR), na.rm=TRUE)/DownsideDeviation(R, MAR)
        SR
    }
    
    # Checking input if SE = TRUE
    if(SE){
      SE.check <- TRUE
      if(!requireNamespace("RPESE", quietly = TRUE)){
        warning("Package \"RPESE\" needed for standard errors computation. Please install it.",
                call. = FALSE)
        SE <- FALSE
      }
    }
    
    # SE Computation
    if(isTRUE(SE)){
      
      # Setting the control parameters
      if(is.null(SE.control))
        SE.control <- RPESE.control(estimator="SoR")

      # Computation of SE (optional)
      ses=list()
      # For each of the method specified in se.method, compute the standard error
      for(mymethod in SE.control$se.method){
        ses[[mymethod]]=RPESE::EstimatorSE(R, estimator.fun = "SoR", se.method = mymethod, 
                                           cleanOutliers=SE.control$cleanOutliers,
                                           fitting.method=SE.control$fitting.method,
                                           freq.include=SE.control$freq.include,
                                           freq.par=SE.control$freq.par,
                                           a=SE.control$a, b=SE.control$b,
                                           threshold = "const", const=MAR,
                                           ...)
        ses[[mymethod]]=ses[[mymethod]]$se
      }
      ses <- t(data.frame(ses))
    }

    # apply across multi-column data if we have it
    result = apply(R, MARGIN = 2, sr, MAR = MAR)
    dim(result) = c(1,NCOL(R))
    colnames(result) = colnames(R)
    rownames(result) = paste("Sortino Ratio (MAR = ", round(mean(MAR)*100,3),"%)", sep="")
    
    if(SE) # Check if SE computation
      return(rbind(result, ses)) else
        return (result)
}
```

### R omega ratio

[src](https://github.com/braverock/PerformanceAnalytics/blob/master/R/Omega.R)

Keating and Shadwick (2002) proposed Omega (referred to as Gamma in their
original paper) as a way to capture all of the higher moments of the returns
distribution.

Mathematically, Omega is: integral[L to b](1 - F(r))dr / integral[a to
L](F(r))dr

where the cumulative distribution F is defined on the interval (a,b). L is
the loss threshold that can be specified as zero, return from a benchmark
index, or an absolute rate of return - any specified level. When comparing
alternatives using Omega, L should be common.

Input data can be transformed prior to calculation, which may be useful for
introducing risk aversion.

This function returns a vector of Omega, useful for plotting.  The steeper,
the less risky.  Above it's mean, a steeply sloped Omega also implies a very
limited potential for further gain.

Omega has a value of 1 at the mean of the distribution.

Omega is sub-additive.  The ratio is dimensionless.

Kazemi, Schneeweis, and Gupta (2003), in "Omega as a Performance Measure"
show that Omega can be written as: Omega(L) = C(L)/P(L) where C(L) is
essentially the price of a European call option written on the investment
and P(L) is essentially the price of a European put option written on the
investment.  The maturity for both options is one period (e.g., one month)
and L is the strike price of both options.

The numerator and the denominator can be expressed as:

exp(-Rf=0)*E[max(x - L, 0)] exp(-Rf=0)*E[max(L - x, 0)] with exp(-Rf=0) calculating the

present values of the two, where rf is the per-period riskless rate.

The first three methods implemented here focus on that observation. The
first method takes the simplification described above.  The second uses the
Black-Scholes option pricing as implemented in fOptions.  The third uses the
binomial pricing model from fOptions.  The second and third methods are not
implemented here.

The fourth method, "interp", creates a linear interpolation of the cdf of
returns, calculates Omega as a vector, and finally interpolates a function
for Omega as a function of L.  This method requires library `Hmisc`,
which can be found on CRAN.

- param L L is the loss threshold that can be specified as zero, return from a benchmark index, or an absolute rate of return - any specified level
- param method one of: simple, interp, binomial, blackscholes
- param output one of: point (in time), or full (distribution of Omega)
- param Rf risk free rate, as a single number

references

Keating, J. and Shadwick, W.F. The Omega Function. working paper.
Finance Development Center, London. 2002.

Kazemi, Schneeweis, and Gupta. Omega as a Performance Measure. 2003.

```R
#' calculate Omega for a return series
#' 
#' Keating and Shadwick (2002) proposed Omega (referred to as Gamma in their
#' original paper) as a way to capture all of the higher moments of the returns
#' distribution.
#' 
#' Mathematically, Omega is: integral[L to b](1 - F(r))dr / integral[a to
#' L](F(r))dr
#' 
#' where the cumulative distribution F is defined on the interval (a,b). L is
#' the loss threshold that can be specified as zero, return from a benchmark
#' index, or an absolute rate of return - any specified level. When comparing
#' alternatives using Omega, L should be common.
#' 
#' Input data can be transformed prior to calculation, which may be useful for
#' introducing risk aversion.
#' 
#' This function returns a vector of Omega, useful for plotting.  The steeper,
#' the less risky.  Above it's mean, a steeply sloped Omega also implies a very
#' limited potential for further gain.
#' 
#' Omega has a value of 1 at the mean of the distribution.
#' 
#' Omega is sub-additive.  The ratio is dimensionless.
#' 
#' Kazemi, Schneeweis, and Gupta (2003), in "Omega as a Performance Measure"
#' show that Omega can be written as: Omega(L) = C(L)/P(L) where C(L) is
#' essentially the price of a European call option written on the investment
#' and P(L) is essentially the price of a European put option written on the
#' investment.  The maturity for both options is one period (e.g., one month)
#' and L is the strike price of both options.
#' 
#' The numerator and the denominator can be expressed as: exp(-Rf=0) * E[max(x
#' - L, 0)] exp(-Rf=0) * E[max(L - x, 0)] with exp(-Rf=0) calculating the
#' present values of the two, where rf is the per-period riskless rate.
#' 
#' The first three methods implemented here focus on that observation. The
#' first method takes the simplification described above.  The second uses the
#' Black-Scholes option pricing as implemented in fOptions.  The third uses the
#' binomial pricing model from fOptions.  The second and third methods are not
#' implemented here.
#' 
#' The fourth method, "interp", creates a linear interpolation of the cdf of
#' returns, calculates Omega as a vector, and finally interpolates a function
#' for Omega as a function of L.  This method requires library \code{Hmisc},
#' which can be found on CRAN.
#' 
#' @param R an xts, vector, matrix, data frame, timeSeries or zoo object of
#' asset returns
#' @param L L is the loss threshold that can be specified as zero, return from
#' a benchmark index, or an absolute rate of return - any specified level
#' @param method one of: simple, interp, binomial, blackscholes
#' @param output one of: point (in time), or full (distribution of Omega)
#' @param Rf risk free rate, as a single number
#' @param SE TRUE/FALSE whether to ouput the standard errors of the estimates of the risk measures, default FALSE.
#' @param SE.control Control parameters for the computation of standard errors. Should be done using the \code{\link{RPESE.control}} function.
#' @param \dots any other passthru parameters
#' @author Peter Carl
#' @seealso \code{\link[Hmisc]{Ecdf}}
#' @references Keating, J. and Shadwick, W.F. The Omega Function. working
#' paper. Finance Development Center, London. 2002. Kazemi, Schneeweis, and
#' Gupta. Omega as a Performance Measure. 2003.
###keywords ts multivariate distribution models
#' @examples
#' 
#'     data(edhec)
#'     Omega(edhec)
#'     Omega(edhec[,13],method="interp",output="point")
#'     Omega(edhec[,13],method="interp",output="full")
#' 
#' @export
Omega <-
function(R, L = 0, method = c("simple", "interp", "binomial", "blackscholes"), 
         output = c("point", "full"), Rf = 0, 
         SE=FALSE, SE.control=NULL,
         ...)
{ # @author Peter Carl

    # DESCRIPTION
    # Keating and Shadwick (2002) proposed Omega (referred to as Gamma in their
    # original paper) as a way to capture all of the higher moments of the
    # returns distribution.  Mathematically, Omega is:
    #   integral[L to b](1 - F(r))dr / integral[a to L](F(r))dr
    # where the cumulative distribution F is defined on the interval (a,b).
    # L is the loss threshold that can be specified as zero, return from a
    # benchmark index, or an absolute rate of return - any specified level.
    # When comparing alternatives using Omega, L should be common.  Input data
    # can be transformed prior to calculation, which may be useful for
    # introducing risk aversion.

    # This function returns a vector of Omega, useful for plotting.  The
    # steeper, the less risky.  Above it's mean, a steeply sloped Omega also
    # implies a very limited potential for further gain.

    # Omega has a value of 1 at the mean of the distribution.

    # Omega is sub-additive.  The ratio is dimensionless.

    # Kazemi, Schneeweis, and Gupta (2003), in "Omega as a Performance Measure"
    # shows that Omega can be written as:
    #   Omega(L) = C(L)/P(L)
    # where C(L) is essentially the price of a European call option written
    # on the investment and P(L) is essentially the price of a European put
    # option written on the investment.  The maturity for both options is
    # one period (e.g., one month) and L is the strike price of both options.

    # The numerator and the denominator can be expressed as:
    #   exp(-Rf) * E[max(x - L, 0)]
    #   exp(-Rf) * E[max(L - x, 0)]
    # with exp(-Rf) calculating the present values of the two, where Rf is
    # the per-period riskless rate.

    # The first three methods implemented here focus on that observation.
    # The first method takes the simplification described above.  The second
    # uses the Black-Scholes option pricing as implemented in fOptions.  The
    # third uses the binomial pricing model from fOptions.  The second and
    # third methods are not implemented here.

    # The fourth method, "interp", creates a linear interpolation of the cdf of
    # returns, calculates Omega as a vector, and finally interpolates a function
    # for Omega as a function of L.

    # FUNCTION
    method = method[1]
    output = output[1]
    
    # Checking input if SE = TRUE
    if(SE){
      SE.check <- TRUE
      if(!requireNamespace("RPESE", quietly = TRUE)){
        warning("Package \"RPESE\" needed for standard errors computation. Please install it.",
                call. = FALSE)
        SE <- FALSE
      }
      if(!(method %in% c("simple"))){
        warning("To return SEs, \"method\" must be \"simple\".",
                call. = FALSE)
        SE.check <- FALSE
      }
      if(!(output %in% c("point"))){
        warning("To return SEs, \"output\" must be \"point\".",
                call. = FALSE)
        SE.check <- FALSE
      }
    }
    
    # SE Computation
    if(SE){

      # Setting the control parameters
      if(is.null(SE.control))
        SE.control <- RPESE.control(estimator="OmegaRatio")
      
      # Computation of SE (optional)
      ses=list()
      # For each of the method specified in se.method, compute the standard error
      for(mymethod in SE.control$se.method){
        ses[[mymethod]]=RPESE::EstimatorSE(R, estimator.fun = "OmegaRatio", se.method = mymethod, 
                                           cleanOutliers=SE.control$cleanOutliers,
                                           fitting.method=SE.control$fitting.method,
                                           freq.include=SE.control$freq.include,
                                           freq.par=SE.control$freq.par,
                                           a=SE.control$a, b=SE.control$b,
                                           const = L,
                                           ...)
        ses[[mymethod]]=ses[[mymethod]]$se
      }
      ses <- t(data.frame(ses))
      # Removing SE output if inappropriate arguments
      if(!SE.check){
        ses.rownames <- rownames(ses)
        ses.colnames <- colnames(ses)
        ses <- matrix(NA, nrow=nrow(ses), ncol=ncol(ses))
        rownames(ses) <- ses.rownames
        colnames(ses) <- ses.colnames
      }
    }

    if (is.vector(R)) {
        x = na.omit(R)

        switch(method,
            simple = {
                numerator = exp(-Rf) * mean(pmax(x - L, 0))
                denominator = exp(-Rf) * mean(pmax(L - x, 0))
                omega = numerator/denominator
            },
            binomial = {
                warning("binomial method not yet implemented, using interp")
                method = "interp"
            },
            blackscholes = {
                warning("blackscholes method not yet implemented, using interp")
                method = "interp"
            },
            interp = {

                stopifnot(requireNamespace("Hmisc",quietly=TRUE))
                a = min(x)
                b = max(x)

                xcdf = Hmisc::Ecdf(x, pl=FALSE)
                f <- approxfun(xcdf$x,xcdf$y,method="linear",ties="ordered")

                if(output == "full") {
                    omega = as.matrix(cumsum(1-f(xcdf$x))/cumsum(f(xcdf$x)))
                    names(omega) = xcdf$x
                }
                else {
                # returns only the point value for L
                    # to get a point measure for omega, have to interpolate
                    omegafull = cumsum(1-f(xcdf$x))/cumsum(f(xcdf$x)) # ????????
                    g <- approxfun(xcdf$x,omegafull,method="linear",ties="ordered")
                    omega = g(L)
                }
            }
        ) # end method switch

        result = omega
    }
    else {
        if(length(Rf)>1) Rf<-mean(Rf)
        if(length(L)>1) L<-mean(L)
        
        R = checkData(R, method = "matrix", ... = ...)
        if(output=="full")
            R = R[,1,drop=FALSE] # constrain to one column
        result = apply(R, 2, Omega, L = L, method = method, output = output, Rf = Rf,
            ... = ...)
        if(output!="full") {
            dim(result) = c(1,NCOL(R))
            rownames(result) = paste("Omega (L = ", round(L*100,1),"%)", sep="")
        }
        colnames(result) = colnames(R)
        
        if(SE) # Check if SE computation
          return(rbind(result, ses)) else
            return(result)
    }
}
```

### R Adjusted Sharpe ratio of the return distribution

Adjusted Sharpe ratio Bacon2008[^zadjsr_Bacon2008] [Bacon2008](#adjsr_Bacon2008) was introduced by Pezier and White (2006)
PezierWhite2006[^zadjsr_PezierWhite2006] [PezierWhite2006](#adjsr_PezierWhite2006) to adjust
for skewness and kurtosis by incorporating a penalty factor for negative skewness
and excess kurtosis.
$$Adjusted Sharpe Ratio = SR * [1 + (\frac{S}{6}) * SR - (\frac{K - 3}{24}) * SR^2]$$
where $SR$ is the Sharpe ratio with data annualized, $S$ is the skewness and $K$ is the kurtosis

References

<!-- markdownlint-disable no-inline-html -->
<a id="adjsr_Bacon2008"></a>
<!-- markdownlint-restore -->
Carl Bacon, **Practical portfolio performance measurement and attribution**, second edition 2008 p.99

<!-- markdownlint-disable no-inline-html -->
<a id="adjsr_PezierWhite2006"></a>
<!-- markdownlint-restore -->
Pezier, Jaques and White, Anthony. 2006. **The Relative Merits of Investable Hedge Fund Indices and of Funds of Hedge Funds in Optimal Passive Portfolios.**
[url](http://econpapers.repec.org/paper/rdgicmadp/icma-dp2006-10.htm)

[^zadjsr_Bacon2008]: My reference, with further explanation and a [supporting link](https://website.com).

[^zadjsr_PezierWhite2006]: Another reference.

```R
AdjustedSharpeRatio <- function (R, Rf = 0, ...)
{
  if (ncol(R)==1 || is.null(R) || is.vector(R)) {
    R = na.omit(R)
    if(length(R)<2) return(NA)
    SR = SharpeRatio.annualized(R, Rf, ...)
    K = kurtosis(R, method = "moment")
    S = skewness(R)
    result = SR*(1+(S/6)*SR-((K-3)/24)*SR^2)
    return(result)
  } else {
    result = apply(R, MARGIN = 2, AdjustedSharpeRatio, Rf = Rf, ...)
    result<-t(result)
    colnames(result) = colnames(R)
    rownames(result) = paste("Adjusted Sharpe ratio (Risk free = ",Rf,")", sep="")
    return(result)
  }
}
```

### R Sharpe

Calculate a traditional or modified Sharpe Ratio of Return over StdDev or VaR or ES

The Sharpe ratio is simply the return per unit of risk (represented by
variability).  In the classic case, the unit of risk is the standard
deviation of the returns.
$$\frac{\overline{(R_{a}-R_{f})}}{\sqrt{\sigma_{(R_{a}-R_{f})}}}$$
William Sharpe now recommends`InformationRatio` preferentially
to the original Sharpe Ratio.

The higher the Sharpe ratio, the better the combined performance of "risk" and return.

As noted, the traditional Sharpe Ratio is a risk-adjusted measure of return
that uses standard deviation to represent risk.

A number of papers now recommend using a "modified Sharpe" ratio using a
Modified Cornish-Fisher VaR or CVaR/Expected Shortfall as the measure of Risk.

We have extended this concept to create multivariate modified
Sharpe-like Ratios for standard deviation, Gaussian VaR, modified VaR,
Gaussian Expected Shortfall, and modified Expected Shortfall. See
`VaR` and `ES`.  You can pass additional arguments
to `VaR` and `ES` via `...` The most important is
probably the 'method' argument

Most recently, we have added Downside Sharpe Ratio (DSR) (see `DownsideSharpeRatio`),
a short name for what Ziemba (2005) called the "Symmetric Downside Risk Sharpe Ratio"
and is defined as the ratio of the mean return  to the square root of lower semivariance:
$$\frac{\overline{(R_{a}-R_{f})}}{\sqrt{2}SemiSD(R_a)}$$
This function returns a traditional or modified Sharpe ratio for the same
periodicity of the data being input (e.g., monthly data -> monthly SR)

The Sharpe ratio is simply the return per unit of risk (represented by
variability).  The higher the Sharpe ratio, the better the combined
performance of "risk" and return.

The Sharpe Ratio is a risk-adjusted measure of return that uses
standard deviation to represent risk.

A number of papers now recommend using a "modified Sharpe" ratio
using a Modified Cornish-Fisher VaR as the measure of Risk.

- param Rf risk free rate, in same period as your returns
- param p confidence level for calculation, default p=.95
- param FUN one of "StdDev" or "VaR" or "ES" to use as the denominator

references

Sharpe, W.F. The Sharpe Ratio,\emph{Journal of Portfolio Management},Fall 1994, 49-58.

Laurent Favre and Jose-Antonio Galeano. Mean-Modified Value-at-Risk Optimization with Hedge Funds.
Journal of Alternative Investment, Fall 2002, v 5.

Ziemba, W. T. (2005). The symmetric downside-risk Sharpe ratio.
The Journal of  Portfolio Management, 32(1), 108-122.

```R
#' SharpeRatio(managers[,1,drop=FALSE], Rf=.035/12, FUN="StdDev") 
#' SharpeRatio(managers[,1,drop=FALSE], Rf = managers[,10,drop=FALSE], FUN="StdDev")
#' SharpeRatio(managers[,1:6], Rf=.035/12, FUN="StdDev") 
#' SharpeRatio(managers[,1:6], Rf = managers[,10,drop=FALSE], FUN="StdDev")

SharpeRatio <-
function (R, Rf = 0, p = 0.95, FUN=c("StdDev", "VaR","ES", "SemiSD"), weights=NULL, annualize = FALSE , 
          SE=FALSE, SE.control=NULL,
          ...)
{
    R = checkData(R)
    
    if(!is.null(dim(Rf)))
        Rf = checkData(Rf)
        
    if(annualize){ # scale the Rf to the periodicity of the calculation
        freq = periodicity(R)
        switch(freq$scale,
            minute = {stop("Data periodicity too high")},
            hourly = {stop("Data periodicity too high")},
            daily = {scale = 252},
            weekly = {scale = 52},
            monthly = {scale = 12},
            quarterly = {scale = 4},
            yearly = {scale = 1}
        )
    } else {
        scale = 1 # won't scale the Rf, will leave it at the same periodicity
    }
    # TODO: Consolidate annualized and regular SR calcs
    srm <-function (R, ..., Rf, p, FUNC)
    {
        FUNCT <- match.fun(FUNC)
        xR = Return.excess(R, Rf)
        SRM = mean(xR, na.rm=TRUE)/FUNCT(R=R, p=p, ...=..., invert=FALSE)
        SRM
    }
    sra <-function (R, ..., Rf, p, FUNC)
    {
        if(FUNC == "StdDev") {
            risk <- StdDev.annualized(x=R, ...)
        } else {
            FUNCT <- match.fun(FUNC)
            risk <- FUNCT(R=R, p=p, ...=..., invert=FALSE)
        }
        xR = Return.excess(R, Rf)
        SRA = Return.annualized(xR)/risk
        SRA
    }
    
    i=1
    if(is.null(weights)){
        result = matrix(nrow=length(FUN), ncol=ncol(R)) 
        colnames(result) = colnames(R) 
    } 
    else {
        result = matrix(nrow=length(FUN))
    }
    
    tmprownames=vector()
    
    # Checking input if SE = TRUE
    if(SE){
      
      if(!requireNamespace("RPESE", quietly = TRUE)){
        warning("Package \"RPESE\" needed for standard errors computation. Please install it.",
                call. = FALSE)
        SE <- FALSE
      }
      ses.full <- matrix(ncol=ncol(R), nrow=0)
    }
    
    result.final <- matrix(nrow=ncol(R))
    for(FUNCT in FUN){
      
      # Setting the measure
      if(FUNCT=="StdDev")
        SR.measure <- "SR" else if(FUNCT=="ES")
          SR.measure <- "ESratio" else if(FUNCT=="VaR")
            SR.measure <- "VaRratio" else if(FUNCT=="SemiSD")
              SR.measure <- "DSR"
    
      # SE Computation
      if(SE){
          
          # Setting the control parameters
          if(is.null(SE.control))
            SE.control <- RPESE.control(estimator=SR.measure)
          
          # Computation of SE (optional)
          ses=list()
          # For each of the method specified in se.method, compute the standard error
          for(mymethod in SE.control$se.method){
            ses[[mymethod]]=RPESE::EstimatorSE(R, estimator.fun = SR.measure, se.method = mymethod, 
                                               cleanOutliers=SE.control$cleanOutliers,
                                               fitting.method=SE.control$fitting.method,
                                               freq.include=SE.control$freq.include,
                                               freq.par=SE.control$freq.par,
                                               a=SE.control$a, b=SE.control$b,
                                               alpha=1-p, rf=Rf, 
                                               ...)
            ses[[mymethod]]=ses[[mymethod]]$se
          }
          ses.full <- rbind(ses.full, t(data.frame(ses)))
        }
    
        if(FUNCT=="SemiSD")
          result[i,] <- DownsideSharpeRatio(R, rf=Rf, ...) else{
            
          if (is.null(weights)){
              if(annualize)
                  result[i,] = sapply(R, FUN=sra, Rf=Rf, p=p, FUNC=FUNCT, ...)
              else
                  result[i,] = sapply(R, FUN=srm, Rf=Rf, p=p, FUNC=FUNCT, ...)
          }
          else { # TODO FIX this calculation, currently broken
              result[i,] = mean(R%*%weights,na.rm=TRUE)/match.fun(FUNCT)(R, Rf=Rf, p=p, weights=weights, portfolio_method="single", ...=...)
          }
        }
        tmprownames = c(tmprownames, paste(if(annualize) "Annualized ", FUNCT, " Sharpe", " (Rf=", round(scale*mean(Rf)*100,1), "%, p=", round(p*100,1),"%):", sep=""))

        i=i+1 #increment counter
    }
    rownames(result)=tmprownames
    
    if(SE)
      return(rbind(result, ses.full)) else
        return(result)
}

#' @export
#' @rdname SharpeRatio
SharpeRatio.modified <-
function (R, Rf = 0, p = 0.95, FUN=c("StdDev", "VaR","ES"), weights=NULL, ...) {
    .Deprecated("SharpeRatio", package="PerformanceAnalytics", "The SharpeRatio.modified function has been deprecated in favor of a newer SharpeRatio wrapper that will cover both the classic case and a larger suite of modified Sharpe Ratios.  This deprecated function may be removed from future versions")

    return(SharpeRatio(R = R, Rf = Rf, p = p, FUN = FUN, weights=weights, ...))
}
```

### R Downside Sharpe Ratio

[src](https://github.com/braverock/PerformanceAnalytics/blob/master/R/DownsideSharpeRatio.R)

The Downside Sharpe Ratio (DSR) is a short name for what Ziemba (2005)
called the "Symmetric Downside Risk Sharpe Ratio" and is defined as the
ratio of the mean excess return to the square root of lower semivariance:

$$\frac{\overline{(R_{a}-R_{f})}}{\sqrt{2}SemiSD(R_a)}$$

- param rf Risk-free interest rate.

Ziemba, W. T. (2005). The symmetric downside-risk Sharpe ratio. The Journal of Portfolio Management, 32(1), 108-122.

```R
DownsideSharpeRatio <- function(R, rf=0, SE=FALSE, SE.control=NULL, ...){
  R = checkData(R, method="matrix")
  
  # Adjusting the returns if rf is a vector of same length
  if(length(rf)>1){
    R <- apply(R, 2, function(x, rf) return(x-as.numeric(rf)), rf=rf)
    rf <- 0
  }
  
  # Downside SR
  DSR <- function(returns, rf = 0){
    
    # Computing the mean of the returns
    mu.hat <- mean(returns)
    # Computing the SemiSD
    semisd <- sqrt((1/length(returns))*sum((returns-mu.hat)^2*(returns <= mu.hat)))
    # Computing the SemiMean
    semimean <- (1/length(returns))*sum((returns-mu.hat)*(returns <= mu.hat))
    # Computing DSR of the returns
    DSR <- (mu.hat-rf)/(semisd*sqrt(2))
    
    # Returning estimate
    return(DSR)
  }
  # Computation of Rachev Ratio
  myDSR <- t(apply(R, 2, DSR, rf=rf))
  rownames(myDSR) <- "Downside Sharpe Ratio"
}
```

### R Bernardo and Ledoit ratio of the return distribution

[src](https://github.com/braverock/PerformanceAnalytics/blob/master/R/BernadoLedoitratio.R)

To calculate Bernardo and Ledoit ratio we take the sum of the subset of
returns that are above 0 and we divide it by the opposite of the sum of
the subset of returns that are below 0
$$BernardoLedoitRatio(R) = \frac{\frac{1}{n}\sum^{n}_{t=1}{max(R_{t},0)}}{\frac{1}{n}\sum^{n}_{t=1}{max(-R_{t},0)}}$$
where $n$ is the number of observations of the entire series

references

Carl Bacon, \emph{Practical portfolio performance measurement and attribution}, second edition 2008 p.95

```R
BernardoLedoitRatio <- function (R, ...)
{
    R <- checkData(R)
    if (ncol(R)==1 || is.null(R) || is.vector(R)) {
       R = na.omit(R)
       r1 = R[which(R > 0)]
       r2 = R[which(R < 0)]
       result = sum(r1)/-sum(r2)
       return(result)
    }  
    else {
        result = apply(R, MARGIN = 2, BernardoLedoitRatio, ...)
        result<-t(result)
        colnames(result) = colnames(R)
        rownames(result) = paste("Bernardo and Ledoit ratio", sep="")
        return(result)
    }
}
```

### R Burke ratio of the return distribution

[src](https://github.com/braverock/PerformanceAnalytics/blob/master/R/BurkeRatio.R)

To calculate Burke ratio we take the difference between the portfolio
return and the risk free rate and we divide it by the square root of the
sum of the square of the drawdowns. To calculate the modified Burke ratio
we just multiply the Burke ratio by the square root of the number of datas.

$$Burke Ratio = \frac{r_P - r_F}{\sqrt{\sum^{d}_{t=1}{D_t}^2}}$$
$$Modified Burke Ratio = \frac{r_P - r_F}{\sqrt{\sum^{d}_{t=1}\frac{{D_t}^2}{n}}}$$

where $n$ is the number of observations of the entire series, $d$ is number of drawdowns,
$r_P$ is the portfolio return, $r_F$ is the risk free rate and $D_t$ the $t^{th}$ drawdown.

- param Rf the risk free rate
- param modified a boolean to decide which ratio to calculate between Burke ratio and modified Burke ratio.

references

Carl Bacon, **Practical portfolio performance measurement and attribution**, second edition 2008 p.90-91

```R
BurkeRatio <- function (R, Rf = 0, modified = FALSE, ...)
{
    drawdown = c()
    R0 <- R
    R = checkData(R, method="matrix")
    if (ncol(R)==1 || is.null(R) || is.vector(R)) {
       calcul = FALSE
       n = length(R)

       number_drawdown = 0
       in_drawdown = FALSE
       peak = 1

        for (i in (1:length(R))) {
            if (!is.na(R[i])) {
             calcul = TRUE
            }
        }

       if(!calcul) {
            result = NA
       }
       else
       {
         period = Frequency(R)
         R = na.omit(R)
         for (i in (2:length(R))) {
          if (R[i]<0)
          {
            if (!in_drawdown)
            {
              peak = i-1
              number_drawdown = number_drawdown + 1
              in_drawdown = TRUE
            }
          }
          else
          {
            if (in_drawdown)
            {
              temp = 1
              boundary1 = peak+1
              boundary2 = i-1
              for(j in (boundary1:boundary2)) {
                temp = temp*(1+R[j]*0.01)
              }
              drawdown = c(drawdown, (temp - 1) * 100)
              in_drawdown = FALSE
            }
          }
        }
      if (in_drawdown)
      {
          temp = 1
    boundary1 = peak+1
    boundary2 = i
          for(j in (boundary1:boundary2)) {
         temp = temp*(1+R[j]*0.01)
    }
    drawdown = c(drawdown, (temp - 1) * 100)
    in_drawdown = FALSE
      }

      D = Drawdowns(R)

       Rp = (prod(1 + R))^(period / length(R)) - 1
         result = (Rp - Rf)/sqrt(sum(drawdown^2))
       if(modified)
       {
    result = result * sqrt(n)
       }
}
       return(result)
    }  
    else {
      R = checkData(R)
        result = apply(R, MARGIN = 2, BurkeRatio, Rf = Rf, modified = modified, ...)
        result<-t(result)
        colnames(result) = colnames(R)
  if (modified)
  {
           rownames(result) = paste("Modified Burke ratio (Risk free = ",Rf,")", sep="")
  }
  else
  {
           rownames(result) = paste("Burke ratio (Risk free = ",Rf,")", sep="")
  }
        return(result)
    }
}
```

### R Excess returns

[src](https://github.com/braverock/PerformanceAnalytics/blob/master/R/Return.excess.R)

Calculates the returns of an asset in excess of the given "risk free rate"
for the period.

Ideally, your risk free rate will be for each period you have returns
observations, but a single average return for the period will work too.

Mean of the period return minus the period risk free rate
$$\overline{(R_{a}-R_{f})}$$
OR
mean of the period returns minus a single numeric risk free rate
$$\overline{R_{a}}-R_{f}$$

Note that while we have, in keeping with common academic usage, assumed that
the second parameter will be a risk free rate, you may also use any other
timeseries as the second argument.  A common alteration would be to use a
benchmark to produce excess returns over a specific benchmark, as
demonstrated in the examples below.

- param Rf risk free rate, in same period as your returns, or as a single digit average

references

Bacon, Carl. **Practical Portfolio Performance Measurement and Attribution**. Wiley. 2004. p. 47-52

```R
Return.excess <-
function (R, Rf = 0)
{ # @author Peter Carl
    # Transform input data to a timeseries (xts) object
    R = checkData(R)

    # if the risk free rate is delivered as a timeseries, we'll check it
    # and convert it to an xts object.
    if(!is.null(dim(Rf))){
        Rf = checkData(Rf)
        coln.Rf=colnames(Rf)
        if(is.null(coln.Rf)){
          colnames(Rf) = "Rf"
          coln.Rf = colnames(Rf)
        }
        Rft=cbind(R,Rf)
        Rft=na.locf(Rft[,make.names(coln.Rf)])
        Rf=Rft[which(index(R) %in% index(Rft))]
    }
    else {
        coln.Rf='Rf'
        Rf = reclass(rep(Rf,length(index(R))),R) #patch thanks to Josh to deal w/ TZ issue
    }

    ## prototype
    ## xts(apply(managers[,1:6],2,FUN=function(R,Rf,order.by) {xts(R,order.by=order.by)-Rf}, Rf=xts(managers[,10,drop=F]),order.by=index(managers)),order.by=index(managers))
 
    result = do.call(merge, lapply(1:NCOL(R), function(nc) R[,nc] - coredata(Rf))) # thanks Jeff!
    
    #if (!is.matrix(result)) result = matrix(result, ncol=ncol(R))
    if(!is.null(dim(result))) colnames(result) = paste(colnames(R), ">", coln.Rf)
    #result = reclass(result, R)

    # RESULTS:
    return(result)
}
```

### R calculate a compounded (geometric) cumulative return

[src](https://github.com/braverock/PerformanceAnalytics/blob/master/R/Return.cumulative.R)

This is a useful function for calculating cumulative return over a period of
time, say a calendar year.  Can produce simple or geometric return.

product of all the individual period returns
$$(1+r_{1})(1+r_{2})(1+r_{3})\ldots(1+r_{n})-1=prod(1+R)-1$$

- param geometric utilize geometric chaining (TRUE) or simple/arithmetic chaining (FALSE) to aggregate returns, default TRUE

```R
Return.cumulative <-
function (R, geometric = TRUE)
{ # @author Peter Carl

    # This is a useful function for calculating cumulative return over a period
    # of time, say a calendar year.  Can produce simple or geometric return.

    if (is.vector(R)) {
        R = na.omit(R)
        if (!geometric)
            return(sum(R))
        else {
            return(prod(1+R)-1)
        }
    }
    else {
        R = checkData(R, method = "matrix")
        result = apply(R, 2, Return.cumulative, geometric = geometric)
        dim(result) = c(1,NCOL(R))
        colnames(result) = colnames(R)
        rownames(result) = "Cumulative Return"
        return(result)
    }
}
```

### R Kappa of the return distribution

[src](https://github.com/braverock/PerformanceAnalytics/blob/master/R/Kappa.R)

Introduced by Kaplan and Knowles (2004), Kappa is a generalized
downside risk-adjusted performance measure.

To calculate it, we take the difference of the mean of the distribution
to the target and we divide it by the l-root of the lth lower partial
moment. To calculate the lth lower partial moment we take the subset of
returns below the target and we sum the differences of the target to
these returns. We then return return this sum divided by the length of
the whole distribution.

$$Kappa(R, MAR, l) = \frac{r_{p}-MAR}{\sqrt[l]{\frac{1}{n}*\sum^n_{t=1}
max(MAR-R_{t}, 0)^l}}$$

For l=1 kappa is the Sharpe-omega ratio and for l=2 kappa
is the sortino ratio.

Kappa should only be used to rank portfolios as it is difficult to
interpret the absolute differences between kappas. The higher the
kappa is, the better.

- param MAR Minimum Acceptable Return, in the same periodicity as your returns
- param l the coefficient of the Kappa

references

Carl Bacon, **Practical portfolio performance measurement and attribution**, second edition 2008 p.96

```R
Kappa <- function (R, MAR, l, ...)
{
    R = checkData(R)

    if (ncol(R)==1 || is.null(R) || is.vector(R)) {
       R = na.omit(R)
       r = R[which(R < MAR)]
       n = length(R)
       m = mean(R)
       result = (m-MAR)/(((1/n)*sum((MAR - r)^l))^(1/l))
       return(result)
    }
    else {
        result = apply(R, MARGIN = 2, Kappa, MAR=MAR, l=l, ...)
        result<-t(result)
        colnames(result) = colnames(R)
        rownames(result) = paste("kappa (MAR = ",MAR,"%)", sep="")
        return(result)
    }
}
```

### R  InformationRatio = ActivePremium/TrackingError

[src](https://github.com/braverock/PerformanceAnalytics/blob/master/R/InformationRatio.R)

The Active Premium divided by the Tracking Error.

InformationRatio = ActivePremium/TrackingError

This relates the degree to which an investment has beaten the benchmark to
the consistency with which the investment has beaten the benchmark.

William Sharpe now recommends InformationRatio preferentially to the original SharpeRatio.

references

Sharpe, W.F. The Sharpe Ratio, **Journal of Portfolio Management**, Fall 1994, 49-58.

```R
InformationRatio <-
function (Ra, Rb, scale = NA)
{ # @author Peter Carl

    # DESCRIPTION
    # InformationRatio = ActivePremium/TrackingError

    # FUNCTION
    Ra = checkData(Ra)
    Rb = checkData(Rb)

    Ra.ncols = NCOL(Ra) 
    Rb.ncols = NCOL(Rb)

    pairs = expand.grid(1:Ra.ncols, 1:Rb.ncols)

    if(is.na(scale)) {
        freq = periodicity(Ra)
        switch(freq$scale,
            minute = {stop("Data periodicity too high")},
            hourly = {stop("Data periodicity too high")},
            daily = {scale = 252},
            weekly = {scale = 52},
            monthly = {scale = 12},
            quarterly = {scale = 4},
            yearly = {scale = 1}
        )
    }

    ir <-function (Ra, Rb, scale)
    {
        ap = ActivePremium(Ra, Rb, scale = scale)
        te = TrackingError(Ra, Rb, scale = scale)
        IR = ap/te
        return(IR)
    }

    result = apply(pairs, 1, FUN = function(n, Ra, Rb, scale) ir(Ra[,n[1]], Rb[,n[2]], scale), Ra = Ra, Rb = Rb, scale = scale)

    if(length(result) ==1)
        return(result)
    else {
        result = matrix(result, ncol=Ra.ncols, nrow=Rb.ncols, byrow=TRUE)
        rownames(result) = paste("Information Ratio:", colnames(Rb))
        colnames(result) = colnames(Ra)
        return(result)
    }
}
```

### R Active Premium or Active Return

[src](https://github.com/braverock/PerformanceAnalytics/blob/master/R/ActivePremium.R)

```R
#' Active Premium or Active Return
#'
#' The return on an investment's annualized return minus the benchmark's
#' annualized return.
#'
#' Active Premium = Investment's annualized return - Benchmark's annualized
#' return
#'
#' Also commonly referred to as 'active return'.
#'
#' @param Ra return vector of the portfolio
#' @param Rb return vector of the benchmark asset
#' @param scale number of periods in a year
#'   (daily scale = 252, monthly scale = 12, quarterly scale = 4)
#' @param ... any other passthru parameters to Return.annualized
#'   (e.g., \code{geometric=FALSE})
#' @author Peter Carl
#' @seealso \code{\link{InformationRatio}} \code{\link{TrackingError}}
#'   \code{\link{Return.annualized}}
#' @references Sharpe, W.F. The Sharpe Ratio,\emph{Journal of Portfolio
#'   Management}, Fall 1994, 49-58.
###keywords ts multivariate distribution models
#' @examples
#'
#'     data(managers)
#'     ActivePremium(managers[, "HAM1", drop=FALSE], managers[, "SP500 TR", drop=FALSE])
#'     ActivePremium(managers[,1,drop=FALSE], managers[,8,drop=FALSE])
#'     ActivePremium(managers[,1:6], managers[,8,drop=FALSE])
#'     ActivePremium(managers[,1:6], managers[,8:7,drop=FALSE])
#' @rdname ActivePremium
#' @aliases
#' ActivePremium
#' ActiveReturn
#' @export ActiveReturn ActivePremium
ActiveReturn <- ActivePremium <- function (Ra, Rb, scale = NA, ...)
{ # @author Peter Carl

    # FUNCTION
    Ra = checkData(Ra)
    Rb = checkData(Rb)

    Ra.ncols = NCOL(Ra)
    Rb.ncols = NCOL(Rb)

    pairs = expand.grid(1:Ra.ncols, 1:Rb.ncols)

    if(is.na(scale)) {
        freq = periodicity(Ra)
        switch(freq$scale,
            minute = {stop("Data periodicity too high")},
            hourly = {stop("Data periodicity too high")},
            daily = {scale = 252},
            weekly = {scale = 52},
            monthly = {scale = 12},
            quarterly = {scale = 4},
            yearly = {scale = 1}
        )
    }

    ap <- function (Ra, Rb, scale)
    {
        merged = na.omit(merge(Ra, Rb)) # align
        ap = (Return.annualized(merged[,1], scale = scale, ...)
              - Return.annualized(merged[,2], scale = scale, ...))
        ap
    }

    result = apply(pairs, 1, FUN = function(n, Ra, Rb, scale) ap(Ra[,n[1]], Rb[,n[2]], scale), Ra = Ra, Rb = Rb, scale = scale)

    if(length(result) == 1)
        return(result)
    else {
        dim(result) = c(Ra.ncols, Rb.ncols)
        colnames(result) = paste("Active Premium:", colnames(Rb))
        rownames(result) = colnames(Ra)
        return(t(result))
    }
}
```

[src](https://github.com/braverock/PerformanceAnalytics/blob/master/R/Return.annualized.R)

calculate an annualized return for comparing instruments with different
length history

An average annualized return is convenient for comparing returns.

Annualized returns are useful for comparing two assets.  To do so, you must
scale your observations to an annual scale by raising the compound return to
the number of periods in a year, and taking the root to the number of total
observations:
$$prod(1+R_{a})^{\frac{scale}{n}}-1=\sqrt[n]{prod(1+R_{a})^{scale}}-1$$
where scale is the number of periods in a year, and n is the total number of
periods for which you have observations.

For simple returns (geometric=FALSE), the formula is:
$$\overline{R_{a}} \cdot scale$$

- param geometric utilize geometric chaining (TRUE) or simple/arithmetic chaining (FALSE) to aggregate returns, default TRUE

references

Bacon, Carl. **Practical Portfolio Performance Measurement and Attribution**. Wiley. 2004. p. 6

```R
Return.annualized <-
function (R, scale = NA, geometric = TRUE )
{ # @author Peter Carl

    # Description:

    # An average annualized return is convenient for comparing returns.
    # @todo: This function could be used for calculating geometric or simple
    # returns

    # R = periods under analysis
    # scale = number of periods in a year (daily f = 252, monthly f = 12,
    # quarterly f = 4)

    # arithmetic average: ra = (f/n) * sum(ri)
    # geometric average: rg = product(1 + ri)^(f/n) - 1

    # @todo: don't calculate for returns less than 1 year

    if(!xtsible(R) & is.na(scale))
        stop("'R' needs to be timeBased or xtsible, or scale must be specified." )
    if(is.na(scale)) {
        freq = periodicity(R)
        switch(freq$scale,
               minute = {stop("Data periodicity too high")},
               hourly = {stop("Data periodicity too high")},
               daily = {scale = 252},
               weekly = {scale = 52},
               monthly = {scale = 12},
               quarterly = {scale = 4},
               yearly = {scale = 1}
        )
    }
    
    # FUNCTION:
    if (is.vector(R)) {
        R = checkData (R)
        R = na.omit(R)
        n = length(R)
        #do the correct thing for geometric or simple returns
        if (geometric) {
            # geometric returns
            result = prod(1 + R)^(scale/n) - 1
        } else {
            # simple returns
            result = mean(R) * scale
        }
        result
    }
    else {
        R = checkData(R, method = "xts")
        result = apply(R, 2, Return.annualized, scale = scale, geometric = geometric)
        dim(result) = c(1,NCOL(R))
        colnames(result) = colnames(R)
        rownames(result) = "Annualized Return"
        return(result)
    }
}
```

### R Calculate Tracking Error of returns against a benchmark

[src](https://github.com/braverock/PerformanceAnalytics/blob/master/R/TrackingError.R)

A measure of the unexplained portion of performance relative to a benchmark.

Tracking error is calculated by taking the square root of the average of the
squared deviations between the investment's returns and the benchmark's
returns, then multiplying the result by the square root of the scale of the
returns.
$$TrackingError=\sqrt{\sum\frac{(R_{a}-R_{b})^{2}}{len(R_{a})\sqrt{scale}}}$$

references

Sharpe, W.F. The Sharpe Ratio, **Journal of Portfolio Management**,Fall 1994, 49-58.

```R
TrackingError <-
function (Ra, Rb, scale = NA)
{ # @author Peter Carl

    # DESCRIPTION
    # TrackingError = sqrt(sum(assetReturns.vec - benchmarkReturns.vec)^2 /
    #                   (length(assetReturns.vec) - 1)) * sqrt(scale)

    # Inputs:
    # Outputs:

    # FUNCTION
    Ra = checkData(Ra)
    Rb = checkData(Rb)

    Ra.ncols = NCOL(Ra) 
    Rb.ncols = NCOL(Rb)

    pairs = expand.grid(1:Ra.ncols, 1:Rb.ncols)

    if(is.na(scale)) {
        freq = periodicity(Ra)
        switch(freq$scale,
            minute = {stop("Data periodicity too high")},
            hourly = {stop("Data periodicity too high")},
            daily = {scale = 252},
            weekly = {scale = 52},
            monthly = {scale = 12},
            quarterly = {scale = 4},
            yearly = {scale = 1}
        )
    }

    te <-function (Ra, Rb, scale)
    {
        TE = sd.xts(Return.excess(Ra, Rb), na.rm=TRUE) * sqrt(scale)
        return(TE)
    }

    result = apply(pairs, 1, FUN = function(n, Ra, Rb, scale) te(Ra[,n[1]], Rb[,n[2]], scale), Ra = Ra, Rb = Rb, scale = scale)

    if(length(result) ==1)
        return(result)
    else {
        dim(result) = c(Ra.ncols, Rb.ncols)
        colnames(result) = paste("Tracking Error:", colnames(Rb))
        rownames(result) = colnames(Ra)
        return(t(result))
    }
}
```

### R Kurtosis

[src](https://github.com/braverock/PerformanceAnalytics/blob/master/R/kurtosis.R)

compute kurtosis of a univariate distribution

This function was ported from the RMetrics package fUtilities to eliminate a
dependency on fUtilties being loaded every time.  This function is identical
except for the addition of \code{\link{checkData}} and additional labeling.

$$Kurtosis(moment) = \frac{1}{n}*\sum^{n}_{i=1}(\frac{r_i - \overline{r}}{\sigma_P})^4$$

$$Kurtosis(excess) = \frac{1}{n}*\sum^{n}_{i=1}(\frac{r_i - \overline{r}}{\sigma_P})^4 - 3$$

$$Kurtosis(sample) =  \frac{n*(n+1)}{(n-1)*(n-2)*(n-3)}*\sum^{n}_{i=1}(\frac{r_i - \overline{r}}{\sigma_{S_P}})^4$$

$$Kurtosis(fisher) = \frac{(n+1)*(n-1)}{(n-2)*(n-3)}*(\frac{\sum^{n}_{i=1}\frac{(r_i)^4}{n}}{(\sum^{n}_{i=1}(\frac{(r_i)^2}{n})^2} - \frac{3*(n-1)}{n+1})$$

$$Kurtosis(sample excess) =  \frac{n*(n+1)}{(n-1)*(n-2)*(n-3)}*\sum^{n}_{i=1}(\frac{r_i - \overline{r}}{\sigma_{S_P}})^4  - \frac{3*(n-1)^2}{(n-2)*(n-3)}$$

where $n$ is the number of return, $\overline{r}$ is the mean of the return
distribution, $\sigma_P$ is its standard deviation and $\sigma_{S_P}$ is its
sample standard deviation.

- param method a character string which specifies the method of computation.
These are either `'moment'`, `'fisher'`, or `'excess'`. If
`'excess'` is selected, then the value of the kurtosis is computed by
the `'moment'` method and a value of 3 will be subtracted.

The `'moment'` method is based on the definitions of kurtosis for
distributions; these forms should be used when resampling (bootstrap or
jackknife). The `'fisher'` method correspond to the usual "unbiased"
definition of sample variance, although in the case of kurtosis exact
unbiasedness is not possible.

The `'sample'` method gives the sample kurtosis of the distribution.

references

Carl Bacon, **Practical portfolio performance measurement and attribution**, second edition 2008 p.84-85

```R
kurtosis <-
    function (x, na.rm = FALSE, method = c("excess", "moment", "fisher", "sample", "sample_excess"), ...)
{
    # @author Diethelm Wuertz
    # @author Brian Peterson   (modify for PerformanceAnalytics)

    # Description:
    #   Returns the value of the kurtosis of a distribution function.

    # Details:
    #   Missing values can be handled.

    # FUNCTION:

    # Method:
    method = match.arg(method)

    R=checkData(x,method="matrix")

    columns = ncol(R)
    columnnames=colnames(R)
    # FUNCTION:
    for(column in 1:columns) {
        x = as.vector(na.omit(R[,column]))
        #x = R[,column]

        if (!is.numeric(x)) stop("The selected column is not numeric")

        # Remove NAs:
        if (na.rm) x = x[!is.na(x)]

        # Warnings:
        if (!is.numeric(x) && !is.complex(x) && !is.logical(x)) {
            warning("argument is not numeric or logical: returning NA")
            return(as.numeric(NA))}


        # Kurtosis:
        n = length(x)
        if (is.integer(x)) x = as.numeric(x)
        if (method == "excess") {
            kurtosis = sum((x-mean(x))^4/(var(x)*(n-1)/n)^2)/length(x) - 3
        }
        if (method == "moment") {
            kurtosis = sum((x-mean(x))^4/(var(x)*(n-1)/n)^2)/length(x)
        }
        if (method == "fisher") {
            kurtosis = ((n+1)*(n-1)*((sum(x^4)/n)/(sum(x^2)/n)^2 -
                (3*(n-1))/(n+1)))/((n-2)*(n-3))
        }
    if (method == "sample") {
       kurtosis = sum((x-mean(x))^4/var(x)^2)*n*(n+1)/((n-1)*(n-2)*(n-3))
    }
    if (method == "sample_excess") {
        kurtosis = sum((x-mean(x))^4/var(x)^2)*n*(n+1)/((n-1)*(n-2)*(n-3)) - 3*(n-1)^2/((n-2)*(n-3))
    }
        kurtosis=array(kurtosis)
        if (column==1) {
            #create data.frame
            result=data.frame(kurtosis=kurtosis)
        } else {
            kurtosis=data.frame(kurtosis=kurtosis)
            result=cbind(result,kurtosis)
        }
    } #end columns loop

    if(ncol(result) == 1) {
        # some backflips to name the single column zoo object
        result = as.numeric(result)
    }
    else{
        colnames(result) = columnnames
        rownames(result) = "Excess Kurtosis"
    }
    # Return Value:
    result
}
```

### R xxx1

```R

```

### R xxx2

```R

```

### R xxx3

```R

```

### R xxx4

```R

```

### R xxx5

```R

```

### R xxx6

```R

```

### R xxx7

```R

```

### R xxx8

```R

```

### R xxx9

```R

```

### R xxx10

```R

```

## Comparison table

| my | ranaussi | stuart | R perf analytics |
| --- | --- | --- | --- |
| aaa | aaa | aaa | aaa |
| aaa | sharpe(rf) | sharpe_ratio(er, rf) | aaa |
| aaa | aaa | aaa | AdjustedSharpeRatio(rf) [R](https://github.com/braverock/PerformanceAnalytics/blob/master/R/AdjustedSharpeRatio.R) |
| aaa | smart_sharpe(rf) | aaa | aaa |
| aaa | probabilistic_sharpe_ratio(rf, smart) | aaa | aaa |
| aaa | aaa | conditional_sharpe_ratio(er, rf, alpha) | aaa |
| aaa | sortino(rf) | sortino_ratio(er, rf, target) | SortinoRatio(MAR) |
| aaa | smart_sortino(rf) | aaa | aaa |
| aaa | adjusted_sortino(rf, smart) | aaa | aaa |
| aaa | probabilistic_sortino_ratio(rf, smart) | aaa | aaa |
| aaa | probabilistic_adjusted_sortino_ratio(rf, smart) | aaa | aaa |
| aaa | aaa | kappa_three_ratio(er, rf, target) | aaa |
| aaa | treynor_ratio(rf, benchmark) | treynor_ratio(er, rf, market) | aaa |
| aaa | omega(rf, required_return) | omega_ratio(er, rf, target) | Omega(L) [R](https://github.com/braverock/PerformanceAnalytics/blob/master/R/Omega.R) |
| aaa | aaa | gain_loss_ratio(target) | aaa |
| aaa | aaa | upside_potential_ratio(target) | aaa |
| aaa | aaa | calmar_ratio(er, rf) | aaa |
| aaa | aaa | sterling_ration(er, rf) | aaa |
| aaa | aaa | burke_ratio(er, rf) | aaa |
| aaa | gain_to_pain_ratio(rf) | aaa | aaa |
| aaa | cagr(rf, compaunded) | aaa | aaa |
| aaa | rar(rf) = cagr(rf)/exposure() | aaa | aaa |
| aaa | skew() | aaa | aaa |
| aaa | kurtosis() | aaa | aaa |
| aaa | aaa | aaa | SkewnessKurtosisRatio(margin) |
| aaa | calmar()=cagr()/max_dd() | aaa | aaa |
| aaa | ulcer_index() | aaa | aaa |
| aaa | ulcer_performance_index(rf) | aaa | aaa |
| aaa | serenity_index(rf) | aaa | aaa |
| risk_of_ruin | ror, risk_of_ruin() | aaa | aaa |
| aaa | var, value_at_risk(sigma, confidence) | var(alpha) | aaa |
| aaa | cvar, conditional_value_at_risk(sigma, confidence) | cvar(alpha) | aaa |
| aaa | aaa | excess_var(er, rf, alpha) | aaa |
| aaa | tail_ratio(cutoff) | aaa | aaa |
| aaa | win_loss_ratio, payoff_ratio() | aaa | aaa |
| aaa | profit_ratio() | aaa | aaa |
| aaa | profit_factor() | aaa | aaa |
| aaa | cpc_index() | aaa | aaa |
| aaa | common_sense_ratio() | aaa | aaa |
| aaa | outlier_win_ratio(quantile) | aaa | aaa |
| aaa | outlier_loss_ratio(quantile) | aaa | aaa |
| aaa | recovery_factor(rf) | aaa | aaa |
| risk_return_ratio | risk_return_ratio() | aaa | aaa |
| aaa | max_drawdown() | max_dd()| aaa |
| aaa | aaa | average_dd() | aaa |
| aaa | aaa | average_dd_squared() | aaa |
| aaa | kelly_criterion() | aaa | aaa |
| aaa | r2, r_squared(benchmark) | aaa | aaa |
| aaa | information_ratio(benchmark) | information_ratio(benchmark) | aaa |
| aaa | alpha, beta, greeks(benchmark) | beta(market) | aaa |
| aaa | aaa | modigliani_ratio(er, rf, benchmark) | aaa |
| aaa | aaa | aaa | aaa |
| aaa | aaa | aaa | aaa |
| aaa | aaa | aaa | aaa |
| aaa | aaa | aaa | aaa |
| aaa | aaa | aaa | aaa |
| aaa | aaa | aaa | aaa |
| aaa | aaa | aaa | aaa |
| aaa | aaa | aaa | aaa |
| aaa | aaa | aaa | aaa |
| aaa | aaa | aaa | aaa |

(1) average returns for asset

(2) average returns for risk free
Note: we use average returns for steps (1) and (2) and not compounded returns

(3) calculate excess return for each period in data (i.e. return asset - return risk free)

(4) standard deviation of excess returns (i.e. standard deviation of (3)
Note: on the explanation above my understanding is that this is not beeing computed, only the volatility of asset.

(5) sharpe ratio = [ (1) - (2) ] / (4)

## Irregularity, volatility, risk, and financial market time series

[link](https://www.pnas.org/doi/full/10.1073/pnas.0405168101)

## Some formulae

[from here](https://github.com/zhangchuheng123/iQuant/blob/dc9a02528dace4cde6974e09050184bb1c5fda2e/chapter_5.md)

$$ Total Portfolio Return = r_p = (P_f - P_i) / P_i $$
$$ Annualized Portfolio Return = r_{p, ann} = (1 + r_p)^{\frac{250}{n}} - 1$$
$$ Annualized Portfolio Volatility = \sigma_p = \sigma(R_p) = \sqrt{\dfrac{250}{n-1} \sum_{i=1}^n (R_{p,i} - \bar{R}_p)^2}$$
$$ Annualized Downside Portfolio Volatility = \sigma_{p,d} = \sigma_d(R_p) = \sqrt{\dfrac{250}{n-1} \sum_{i=1}^n I(R_{p,i} < \bar{R}_p)(R_{p,i} - \bar{R}_p)^2} $$
$$I(x) = \begin{cases} 1 & x\text{ is True} \\ 0 & x\text{ is False} \end{cases}$$
$$ Total Benchmark Return = r_M = (I_f - I_i) / I_i $$
$$ Annualized Benchmark Return = r_{M, ann}= (1 + r_M)^{\frac{250}{n}} - 1$$
$$ Annualized Benchmark Volatility = \sigma_M = \sigma(R_M)$$
$$ Annualized Hedged Return = r_{p, hdg, ann} =  r_{p, ann} - r_{M, ann}$$
$$ Annualized Hedged Volatility = \sigma_{p, hdg} = \sigma(R_{p, hdg})$$
$$R_{p, hdg} = R_p - R_M$$
$$\sigma(\cdot)$$
$$ Beta = \beta_p = \dfrac{cov(R_p, R_M)}{var(R_M)} $$
$$ Alpha = \alpha_p =  r_{p, ann} - [r_f + \beta_p (r_{M, ann} - r_f)]$$
$$ Sharpe = (r_{p, ann} - r_f) / \sigma_p $$
$$Sharpe = \dfrac{\mathbb{E}(r_p) - r_f}{var(r_p)}$$
$$ Information Ratio = IR = (r_{p, ann} - r_{M, ann}) / \sigma_{p,hdg}$$
$$ Sortino = (r_{p, ann} - r_f) / \sigma_{p,d} $$
$$ Max Drawdown = MDD = \max_{i<j} \max((P_i - P_j) / P_i, 0)$$
$$P_i = \prod_{j=1}^i (1 + R_i)$$
$$WinningRate = \dfrac{\text{策略交易盈利次数}}{\text{策略交易总次数}}$$
$$\min var(r_p) = \sum_{i=1}^n \sum_{j=1}^n cov(r_i, r_j) \\ s.t. \mathbb{E}(r_p) = \sum_{i=1}^n w_i \mathbb{E}(r_i) \ge \mu, \sum_{i=1}^n w_i = 1$$

## Definitions

| term | definition |
| --- | --- |
| Mean return | is the average realized stock return. |
| Median return | is the return at which half of returns are above and half are below. The median differs from the mean when distributions are not normal. |
| Standard deviation | describes the dispersion of stock returns. If the distribution is normal, stock returns will fall above or below the mean return by one standard deviation about 68% of the time. About 95% of the time, stock returns will fall within two standard deviations of the mean return. |
| Skewness | describes the symmetry of a distribution. A negative skew (left-tailed) implies that negative returns (relative to the mode) are less common but more extreme than positive returns; likewise, a positive skew (right-tailed) implies that positive returns (relative to the mode) are less common but more extreme than negative returns. In other words, for a right-skewed distribution, the likelihood of extreme positive returns is greater than that of extreme negative returns. In finance, securities that exhibit large negative skewness are often avoided because they imply large downside risk. |
| Kurtosis | describes the peakedness of a distribution. Because a normal distribution has a kurtosis of 3, excess kurtosis is often used instead (ie. kurtosis less 3). A distribution that is platykurtic (excess kurtosis < 0) has a low peak and thin tails; one that is leptokurtic (excess kurtosis > 0) has a high peak and fat tails. A distribution that is leptokurtic will exhibit fluctuations of larger magnitude than one platykurtic distribution, rendering the security more risky. From the above, it's clear that distributions which exhibit high standard deviations, negative skewness and high kurtosis are the riskiest investments |
| The Capital Asset Pricing Model (CAPM) | is a commonly used financial pricing model. It calculates a required rate of return for a stock by adding the risk-free rate to the product of the stock's beta and the market risk premium. Unlike many other models, it assumes there is only one type of risk: market risk. |
| The Fama-French 3 factor model | is a financial pricing model. It expands on the CAPM model by adding two other factors to market risk: size risk and value risk. It uses two portfolios to proxy these risks: SMB and HML, respectively. Because the Fama-French model empirically explains the variability of market returns better than the CAPM, it is believed this is the evidence that investors require size and value risk premia (or, at least, whatever these portfolios truly proxy). The SMB (small minus big market cap) and HML (high minus low book-to-value) are long-short portfolios that are long the top third of small/high stocks and short the bottom third. Both these regression models produce $R^2$ results. The $R^2$ value measures to what degree the independent variable (eg. factor returns) explains the variability of the dependent variable (eg. stock returns). It is a proxy for how \"useful\" the alpha and beta values are. For example, a beta value that is high while the $R^2$ is low implies that the beta value is unreliable. As a result, using this beta in a financial pricing model may not produce robust results. |
| The Sharpe ratio | is a commonly used metric to evaluate stock or portfolio performance. It is defined as mean return less the risk free rate, divided by the standard deviation. Put simply, it is how much the stock returns given each additional unit of risk. |
|The Sortino ratio | expands on the Sharpe ratio by recognizing that standard deviation of returns also includes positive returns. Typically, investors are only concerned with downside risk, not upside. As a result, Sortino updates the Sharpe ratio to use not the standard deviation of all returns, but only negative returns. |
