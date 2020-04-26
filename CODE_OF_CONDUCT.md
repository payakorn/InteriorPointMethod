# Contributor Covenant Code of Conduct

## Our Pledge

In the interest of fostering an open and welcoming environment, we as
contributors and maintainers pledge to making participation in our project and
our community a harassment-free experience for everyone, regardless of age, body
size, disability, ethnicity, sex characteristics, gender identity and expression,
level of experience, education, socio-economic status, nationality, personal
appearance, race, religion, or sexual identity and orientation.

## Our Standards

Examples of behavior that contributes to creating a positive environment
include:

* Using welcoming and inclusive language
* Being respectful of differing viewpoints and experiences
* Gracefully accepting constructive criticism
* Focusing on what is best for the community
* Showing empathy towards other community members

Examples of unacceptable behavior by participants include:

* The use of sexualized language or imagery and unwelcome sexual attention or
 advances
* Trolling, insulting/derogatory comments, and personal or political attacks
* Public or private harassment
* Publishing others' private information, such as a physical or electronic
 address, without explicit permission
* Other conduct which could reasonably be considered inappropriate in a
 professional setting

## Our Responsibilities

Project maintainers are responsible for clarifying the standards of acceptable
behavior and are expected to take appropriate and fair corrective action in
response to any instances of unacceptable behavior.

Project maintainers have the right and responsibility to remove, edit, or
reject comments, commits, code, wiki edits, issues, and other contributions
that are not aligned to this Code of Conduct, or to ban temporarily or
permanently any contributor for other behaviors that they deem inappropriate,
threatening, offensive, or harmful.

## Scope

This Code of Conduct applies both within project spaces and in public spaces
when an individual is representing the project or its community. Examples of
representing a project or community include using an official project e-mail
address, posting via an official social media account, or acting as an appointed
representative at an online or offline event. Representation of a project may be
further defined and clarified by project maintainers.

## Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be
reported by contacting the project team at payakornn@gmail.com. All
complaints will be reviewed and investigated and will result in a response that
is deemed necessary and appropriate to the circumstances. The project team is
obligated to maintain confidentiality with regard to the reporter of an incident.
Further details of specific enforcement policies may be posted separately.

Project maintainers who do not follow or enforce the Code of Conduct in good
faith may face temporary or permanent repercussions as determined by other
members of the project's leadership.

## Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage], version 1.4,
available at https://www.contributor-covenant.org/version/1/4/code-of-conduct.html

[homepage]: https://www.contributor-covenant.org

For answers to common questions about this code of conduct, see
https://www.contributor-covenant.org/faq

#' % FIR filter design with Python and SciPy
#' % Matti Pastell
#' % 15th April 2013

#' # Introduction

#' This an example of a script that can be published using
#' [Pweave](http://mpastell.com/pweave). The script can be executed
#' normally using Python or published to HTML with Pweave
#' Text is written in markdown in lines starting with "`#'` " and code
#' is executed and results are included in the published document.
#' The concept is similar to
#' publishing documents with [MATLAB](http://mathworks.com) or using
#' stitch with [Knitr](http://http://yihui.name/knitr/demo/stitch/).

#' Notice that you don't need to define chunk options (see
#' [Pweave docs](http://mpastell.com/pweave/usage.html#code-chunk-options)
#' ),
#' but you do need one line of whitespace between text and code.
#' If you want to define options you can do it on using a line starting with
#' `#+`. just before code e.g. `#+ term=True, caption='Fancy plots.'`. 
#' If you're viewing the HTML version have a look at the
#' [source](FIR_design.py) to see the markup.

#' The code and text below comes mostly
#' from my blog post [FIR design with SciPy](http://mpastell.com/2010/01/18/fir-with-scipy/),
#' but I've updated it to reflect new features in SciPy. 

#' # FIR Filter Design

#' We'll implement lowpass, highpass and ' bandpass FIR filters. If
#' you want to read more about DSP I highly recommend [The Scientist
#' and Engineer's Guide to Digital Signal
#' Processing](http://www.dspguide.com/) which is freely available
#' online.

#' ## Functions for frequency, phase, impulse and step response

#' Let's first define functions to plot filter
#' properties.

from pylab import *
import scipy.signal as signal
    
#Plot frequency and phase response
def mfreqz(b,a=1):
    w,h = signal.freqz(b,a)
    h_dB = 20 * log10 (abs(h))
    subplot(211)
    plot(w/max(w),h_dB)
    ylim(-150, 5)
    ylabel('Magnitude (db)')
    xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    title(r'Frequency response')
    subplot(212)
    h_Phase = unwrap(arctan2(imag(h),real(h)))
    plot(w/max(w),h_Phase)
    ylabel('Phase (radians)')
    xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    title(r'Phase response')
    subplots_adjust(hspace=0.5)

#Plot step and impulse response
def impz(b,a=1):
    l = len(b)
    impulse = repeat(0.,l); impulse[0] =1.
    x = arange(0,l)
    response = signal.lfilter(b,a,impulse)
    subplot(211)
    stem(x, response)
    ylabel('Amplitude')
    xlabel(r'n (samples)')
    title(r'Impulse response')
    subplot(212)
    step = cumsum(response)
    stem(x, step)
    ylabel('Amplitude')
    xlabel(r'n (samples)')
    title(r'Step response')
    subplots_adjust(hspace=0.5)

#' ## Lowpass FIR filter

#' Designing a lowpass FIR filter is very simple to do with SciPy, all you
#' need to do is to define the window length, cut off frequency and the
#' window.

#' The Hamming window is defined as:
#' $w(n) = \alpha - \beta\cos\frac{2\pi n}{N-1}$, where $\alpha=0.54$ and $\beta=0.46$ 

#' The next code chunk is executed in term mode, see the [Python script](FIR_design.py) for syntax.
#' Notice also that Pweave can now catch multiple figures/code chunk.

#+ term=True
n = 61
a = signal.firwin(n, cutoff = 0.3, window = "hamming")
#Frequency and phase response
mfreqz(a)
show()
#Impulse and step response
figure(2)
impz(a)
show()


#' ## Highpass FIR Filter

#' Let's define a highpass FIR filter, if you compare to original blog
#' post you'll notice that it has become easier since 2009. You don't
#' need to do ' spectral inversion "manually" anymore!

n = 101
a = signal.firwin(n, cutoff = 0.3, window = "hanning", pass_zero=False)
mfreqz(a)
show()

#' ## Bandpass FIR filter

#' Notice that the plot has a caption defined in code chunk options.

#+ caption = "Bandpass FIR filter."
n = 1001
a = signal.firwin(n, cutoff = [0.2, 0.5], window = 'blackmanharris', pass_zero = False)
mfreqz(a)
show()




