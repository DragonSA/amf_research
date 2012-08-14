Valuation of Convertible Bonds with Credit Risk
===============================================

Abstract
--------
A convertible bond is a complex derivative that cannot be priced as a simple
combination of bond and stock components.  Convertible bonds can be broken down
as a bond with two embedded options (a put option for the investor and a call
option for the issuer) and an option to convert the bond into stock.  Due to the
multiple continuous options, the pricing of the convertible bond is path
dependent.

This research project explores and implements a binary tree and finite
difference scheme to price the convertible bond, taking into account credit
risk.

Directory Layout
----------------
This repository has the following layout:
 * common/: common objects (graphics) used by the proposal, presentation and
	report
 * presentation/: the presentation (written in LaTex/Beamer) for this project
	(deadline: 2012/11/19)
 * proposal/: the proposal (written in LaTex) for this project
	(deadline: 2012/07/30)
 * references/: a BibTex reference of all references used, and the sources if
	available
 * report/: the research report (written in LaTex) for this project
	(deadline: 2012/11/23)
 * src/: the code written for this project

Source Code
-----------
The source code is distributed under a BSD 2-clause license, see src/COPYRIGHT
for details.

The code is written in Python 2.7, the following software and versions were
used:
 - Python 2.7.3
 - PyPy 1.9.0
 - pmake (as distributed with FreeBSD 9)

Acknowledgements
--------------
 - Tom McWalter (research supervisor)
 - Coenraad Labuschagne (course coordinator)
