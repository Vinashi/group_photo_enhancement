# group_photo_enhancement


"""
We propose a framework for automatic enhancement of group photographs by
facial expression analysis. We are motivated by the observation that group
photos are seldom perfect. Subjects may have inadvertently closed their
eyes, may be looking away or may not be smiling at that moment. Given a set
of photographs of the same group of people, our algorithm uses face landmark
detection to determine the goodness score for each face instance in those
photos. The scoring function is based on feature identifiers for facial
expressions such as eye-closure, smile and face orientation. Given these
scores, a best composite for the set is synthesized by (a) selecting the
photo with best overall score and (b) replacing any low score faces in that
photo with high scoring faces of the same person from other photos using
delaunay triangulation and seamless cloning.
"""
 
