import bibtexparser

def parseBib(bibtex_str=None, bibtex_filepath=None):
    """
    Parses a bibtex string or file and returns a dictionary with the latex \cite-key as the keys and the value is a dict with title and authors as strings.
    """
    if bibtex_str:
        bib = bibtexparser.loads(bibtex_str)
    elif bibtex_filepath:
        bib = bibtexparser.load(open(bibtex_filepath, encoding='utf-8'))
    else:
        print("No input given to the parseBib function. Exiting.")
        return {}
    bib_dict = {}
    for entry in bib.entries:
        bib_dict[entry['ID']] = {'title': entry['title'] if 'title' in entry.keys() else '', 
                                 'author': entry['author'] if 'author' in entry.keys() else ''}
    return bib_dict # Should maybe be written to a text file


# Only for test-example :
if __name__ == "__main__":
    # Example
    tex_cite_id = 'doshi2017towards'                              # The key of a bibtex entry to be used in the latex document
    bibtex_str = r"""                                               # A string with a bibtex entry
dfgdh
fhdf
fd
cergthehtyh


@article{doshi2017towards,
  title={Towards a rigorous science of interpretable machine learning},
  author={Doshi-Velez, Finale and Kim, Been},
  journal={arXiv preprint arXiv:1702.08608},
  year={2017}
}

@article{lipton2016mythos,
  title={The mythos of model interpretability},
  author={Lipton, Zachary C},
  journal={arXiv preprint arXiv:1606.03490},
  year={2016}
}

@inproceedings{ribeiro2016should,
  title={Why should i trust you?: Explaining the predictions of any classifier},
  author={Ribeiro, Marco Tulio and Singh, Sameer and Guestrin, Carlos},
  booktitle={Proceedings of KDD},
  year={2016}
}


@inproceedings{lundberg2017unified,
  title={A unified approach to interpreting model predictions},
  author={Lundberg, Scott M and Lee, Su-In},
  booktitle={Proceedings of NIPS},
  year={2017}
}

@inproceedings{craven1996extracting,
  title={Extracting tree-structured representations of trained networks},
  author={Craven, Mark and Shavlik, Jude W},
  booktitle={Advances in neural information processing systems},
  year={1996}
}

@article{weisberg2013pretense,
  title={Pretense, counterfactuals, and Bayesian causal models: Why what is not real really matters},
  author={Weisberg, Deena S and Gopnik, Alison},
  journal={Cognitive Science},
  year={2013},
  publisher={Wiley Online Library}
}

@inproceedings{beck2009relating,
  title={Relating developments in children\'s counterfactual thinking and executive functions},
  author={Beck, Sarah R and Riggs, Kevin J and Gorniak, Sarah L},
  journal={Thinking \& reasoning},
  year={2009},
  publisher={Taylor \& Francis}
}

@article{buchsbaum2012power,
  title={The power of possibility: Causal learning, counterfactual reasoning, and pretend play},
  author={Buchsbaum, Daphna and Bridgers, Sophie and Weisberg, Deena Skolnick and Gopnik, Alison},
  journal={Philosophical Trans. of  the Royal Soc. B: Biological Sciences},
  year={2012},
  publisher={The Royal Society}
}

@inproceedings{wachter2017counterfactual,
  title={Counterfactual explanations without opening the black box: Automated decisions and the GDPR},
  author={Wachter, Sandra and Mittelstadt, Brent and Russell, Chris},
  year={2017}
}

@article{waters2014grade,
  title={Grade: Machine learning support for graduate admissions},
  author={Waters, Austin and Miikkulainen, Risto},
  journal={AI Magazine},
  volume={35},
  number={1},
  pages={64},
  year={2014}
}

@article{rockoff2011can,
  title={Can you recognize an effective teacher when you recruit one?},
  author={Rockoff, Jonah E and Jacob, Brian A and Kane, Thomas J and Staiger, Douglas O},
  journal={Education finance and Policy},
  volume={6},
  number={1},
  pages={43--74},
  year={2011},
  publisher={MIT Press}
}

@article{andini2017targeting,
  title={Targeting policy-compliers with machine learning: an application to a tax rebate programme in Italy},
  author={Andini, Monica and Ciani, Emanuele and de Blasio, Guido and D'Ignazio, Alessio and Salvestrini, Viola},
  year={2017}
}

@article{athey2017beyond,
  title={Beyond prediction: Using big data for policy problems},
  author={Athey, Susan},
  journal={Science},
  volume={355},
  number={6324},
  pages={483--485},
  year={2017},
  publisher={American Association for the Advancement of Science}
}

@article{dai2015prediction,
  title={Prediction of hospitalization due to heart diseases by supervised learning methods},
  author={Dai, Wuyang and Brisimi, Theodora S and Adams, William G and Mela, Theofanie and Saligrama, Venkatesh and Paschalidis, Ioannis Ch},
  journal={International journal of medical informatics},
  volume={84},
  number={3},
  pages={189--197},
  year={2015},
  publisher={Elsevier}
}

@article{tomsett2018interpretable,
  title={Interpretable to Whom? A Role-based Model for Analyzing Interpretable Machine Learning Systems},
  author={Tomsett, Richard and Braines, Dave and Harborne, Dan and Preece, Alun and Chakraborty, Supriyo},
  journal={arXiv preprint arXiv:1806.07552},
  year={2018}
}

@inproceedings{kusner2017counterfactual,
  title={Counterfactual fairness},
  author={Kusner, Matt J and Loftus, Joshua and Russell, Chris and Silva, Ricardo},
  booktitle={Advances in Neural Information Processing Systems},
  pages={4066--4076},
  year={2017}
}

@inproceedings{nabi2018fair,
  title={Fair inference on outcomes},
  author={Nabi, Razieh and Shpitser, Ilya},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={2018},
  pages={1931},
  year={2018},
  organization={NIH Public Access}
}

@inproceedings{lai+tan:19, 
     author = {Vivian Lai and Chenhao Tan}, 
     title = {On Human Predictions with Explanations and Predictions of Machine Learning Models: A Case Study on Deception Detection}, 
     year = {2019}, 
     booktitle = {Proceedings of FAT*} 
}

@inproceedings{lakkaraju2016interpretable,
  title={Interpretable decision sets: A joint framework for description and prediction},
  author={Lakkaraju, Himabindu and Bach, Stephen H and Leskovec, Jure},
  booktitle={Proc. KDD},
  year={2016},
}

@inproceedings{lou2013accurate,
  title={Accurate intelligible models with pairwise interactions},
  author={Lou, Yin and Caruana, Rich and Gehrke, Johannes and Hooker, Giles},
  booktitle={Proceedings of KDD},
  year={2013},
}

@inproceedings{lou2012intelligible,
  title={Intelligible models for classification and regression},
  author={Lou, Yin and Caruana, Rich and Gehrke, Johannes},
  booktitle={Proceedings of KDD},
  year={2012},
}

@inproceedings{caruana2015intelligible,
  title={Intelligible models for healthcare: Predicting pneumonia risk and hospital 30-day readmission},
  author={Caruana, Rich and Lou, Yin and Gehrke, Johannes and Koch, Paul and Sturm, Marc and Elhadad, Noemie},
  booktitle={Proceedings of KDD},
  year={2015},
}

@inproceedings{kim2016examples,
  title={Examples are not enough, learn to criticize! criticism for interpretability},
  author={Kim, Been and Khanna, Rajiv and Koyejo, Oluwasanmi O},
  booktitle={Proceedings of NIPS},
  year={2016}
} 

@inproceedings{zhou2018interpretable,
  title={Interpretable basis decomposition for visual explanation},
  author={Zhou, Bolei and Sun, Yiyou and Bau, David and Torralba, Antonio},
  booktitle={Proceedings of ECCV},
  year={2018}
}

@inproceedings{zeiler2014visualizing,
  title={Visualizing and understanding convolutional networks},
  author={Zeiler, Matthew D and Fergus, Rob},
  booktitle={Proceedings of ECCV},
  year={2014},
}

@inproceedings{mahendran2015understanding,
  title={Understanding deep image representations by inverting them},
  author={Mahendran, Aravindh and Vedaldi, Andrea},
  booktitle={Proceedings of CVPR},
  year={2015}
}

@inproceedings{russell2019efficient,
  title={Efficient Search for Diverse Coherent Explanations},
  author={Russell, Chris},
  booktitle={Proceedings of FAT*},
  year={2019}
}

@article{kulesza2012determinantal,
  title={Determinantal point processes for machine learning},
  author={Kulesza, Alex and Taskar, Ben and others},
  journal={Foundations and Trends{\textregistered} in Machine Learning},
  volume={5},
  number={2--3},
  pages={123--286},
  year={2012},
  publisher={Now Publishers, Inc.}
}

@inproceedings{kingma2014adam,
  title={Adam: A method for stochastic optimization},
  author={Kingma, Diederik P and Ba, Jimmy},
  booktitle={Proceedings of ICLR},
  year={2015}
}


@article{dressel2018accuracy,
  title={The accuracy, fairness, and limits of predicting recidivism},
  author={Dressel, Julia and Farid, Hany},
  journal={Science advances},
  volume={4},
  number={1},
  pages={eaao5580},
  year={2018},
  publisher={American Association for the Advancement of Science}
}



@article{tan2017distill,
  title={Distill-and-compare: Auditing black-box models using transparent model distillation},
  author={Tan, S and Caruana, R and Hooker, G and Lou, Y},
  year={2017}
}



@misc{whatifurl,
  author = {PAIR},
  title = {{What-If Tool}},
  howpublished = "\url{https://pair-code.github.io/what-if-tool/}",
  year = {2018}, 
}

@misc{propublica-story,
  author = {Angwin, Julia and Larson, Jeff and Mattu, Surya and Kirchner, Lauren},
  title = {{“Machine bias: There’s software used
across the country to predict future criminals. And it’s biased against blacks”}},
  howpublished = "\url{https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing}",
  year = {2016}, 
}


@misc{adult,
author = "Kohavi, Ronny and Becker, Barry",
year = "1996",
title = "{UCI} Machine Learning Repository",
url = "https://archive.ics.uci.edu/ml/datasets/adult",
institution = "University of California, Irvine, School of Information and Computer Sciences" 
}

@misc{adult-cleaning,
  author = { Zhu, Haojun},
  title = {{Predicting Earning Potential using the Adult Dataset}},
  howpublished = "\url{https://rpubs.com/H\_Zhu/235617}",
  year = {2016}, 
}

@misc{lending1,
  author = {JFdarre},
  title = {{Project 1: Lending Club’s data}},
  howpublished = "\url{https://rpubs.com/jfdarre/119147}",
  year = {2015}, 
}

@misc{lending2,
  author = {Davenport, Kevin},
  title = {{Lending Club Data Analysis Revisited with Python}},
  howpublished = "\url{http://kldavenport.com/lending-club-data-analysis-revisted-with-python/}",
  year = {2015}, 
}
@misc{lending-data,
  title = {{Lending Club Statistics}},
  howpublished = "\url{https://www.lendingclub.com/info/download-data.action}",
}



@misc{german,
  title = {{German credit dataset }},
  howpublished = "\url{https://archive.ics.uci.edu/ml/support/statlog+(german+credit+data)}",
  year = {Accessed 2019}, 
}
@article{guidotti2018lore,   title={Local rule-based explanations of black box decision systems},   author={Guidotti, Riccardo and Monreale, Anna and Ruggieri, Salvatore and Pedreschi, Dino and Turini, Franco and Giannotti, Fosca},   journal={arXiv preprint arXiv:1805.10820},   year={2018} }

@article{mccane2008distance,
  title={Distance functions for categorical and mixed variables},
  author={McCane, Brendan and Albert, Michael},
  journal={Pattern Recognition Letters},
  volume={29},
  number={7},
  pages={986--993},
  year={2008},
  publisher={Elsevier}
}

@inproceedings{ziegler2005improving,
  title={Improving recommendation lists through topic diversification},
  author={Ziegler, Cai-Nicolas and McNee, Sean M and Konstan, Joseph A and Lausen, Georg},
  booktitle={Proceedings of the 14th international conference on World Wide Web},
  pages={22--32},
  year={2005},
  organization={ACM}
}

@inproceedings{ekstrand2014user,
  title={User perception of differences in recommender algorithms},
  author={Ekstrand, Michael D and Harper, F Maxwell and Willemsen, Martijn C and Konstan, Joseph A},
  booktitle={Proceedings of the 8th ACM Conference on Recommender systems},
  pages={161--168},
  year={2014},
  organization={ACM}
}

@article{kunaver2017diversity,
  title={Diversity in recommender systems--A survey},
  author={Kunaver, Matev{\v{z}} and Po{\v{z}}rl, Toma{\v{z}}},
  journal={Knowledge-Based Systems},
  volume={123},
  pages={154--162},
  year={2017},
  publisher={Elsevier}
}

@inproceedings{sanderson2009else,
  title={What else is there? search diversity examined},
  author={Sanderson, Mark and Tang, Jiayu and Arni, Thomas and Clough, Paul},
  booktitle={European Conference on Information Retrieval},
  pages={562--569},
  year={2009},
  organization={Springer}
}

@article{scheibehenne2010can,
  title={Can there ever be too many options? A meta-analytic review of choice overload},
  author={Scheibehenne, Benjamin and Greifeneder, Rainer and Todd, Peter M},
  journal={Journal of consumer research},
  volume={37},
  number={3},
  pages={409--425},
  year={2010},
  publisher={The University of Chicago Press}
}

@article{bundorf2010choice,
  title={Choice set size and decision making: the case of Medicare Part D prescription drug plans},
  author={Bundorf, M Kate and Szrek, Helena},
  journal={Medical Decision Making},
  volume={30},
  number={5},
  pages={582--593},
  year={2010},
  publisher={Sage Publications Sage CA: Los Angeles, CA}
}

@book{pearl2009causality,
  title={Causality},
  author={Pearl, Judea},
  year={2009},
  publisher={Cambridge university press}
}
"""
    file_path_bib = 'Other_stuff/1905.07697v2/cf-explain2.bib'      # The path to a .bib file in non-comitted folder
    
    bib_dict_res = parseBib(bibtex_str=bibtex_str)
    #bib_dict_res = parseBib(bibtex_filepath=file_path_bib)
    print(bib_dict_res[tex_cite_id])
    

    # Getting ids with Bib_to_id
    from Bib_to_id import bib_to_id
    
    ids_dict = {}
    for key, value in bib_dict_res.items():
        ids_dict[key] = bib_to_id(value['title'], value['author'])
    print(ids_dict)
    # Finds id
    # 14/48 by only searching with title
    # 13/48 by searching with title and authors
    # 14/48 with title and authors and then just titles