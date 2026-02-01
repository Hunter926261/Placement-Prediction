class PlacementInput:
    def __init__(
        self,
        CGPA,
        Internships,
        Projects,
        Workshops,
        AptitudeTestScore,
        SoftSkillsRating,
        SSC_Marks,
        HSC_Marks
    ):
        self.CGPA = float(CGPA)
        self.Internships = int(Internships)
        self.Projects = int(Projects)
        self.Workshops = int(Workshops)
        self.AptitudeTestScore = float(AptitudeTestScore)
        self.SoftSkillsRating = float(SoftSkillsRating)
        self.SSC_Marks = float(SSC_Marks)
        self.HSC_Marks = float(HSC_Marks)
