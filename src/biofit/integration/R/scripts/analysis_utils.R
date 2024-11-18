# cumNorm <- function(x, p = cumNormStatFast(x)) {
#   normFactors <- calcNormFactors(x, p = p)
#   x$normFactors <- normFactors
#   return(x)
# }
# 
# calcNormFactors <- function(x, p = cumNormStatFast(x)) {
#   xx <- x
#   xx[x == 0] <- NA
#   qs <- rowQuantiles(xx, probs = p, na.rm = TRUE)
#   norm_factors <- apply(xx, 1, function(row, qs) {
#     row <- (row - .Machine$double.eps)
#     sum(row[row <= qs])
#   }, qs)
#   names(norm_factors) <- rownames(x)
#   as.data.frame(norm_factors)
# }
# 
# cumNormMat <- function(x, p = cumNormStatFast(x), sl = 1000) {
#   xx <- x
#   xx[x == 0] <- NA
# 
#   qs <- rowQuantiles(xx, probs = p, na.rm = TRUE)
# 
#   newMat <- apply(xx, 1, function(row, qs) {
#     row <- (row - .Machine$double.eps)
#     sum(row[row <= qs])
#   }, qs)
#   nmat <- sweep(x, 1, newMat / sl, "/")
#   return(nmat)
# }
# 
# cumNormStat <- function(obj, qFlag = TRUE, pFlag = FALSE, rel = .1, ...) {
#   mat <- returnAppropriateObj(obj, FALSE, FALSE)
#   if (any(rowSums(mat) == 0)) stop("Warning: empty feature")
# 
#   smat <- apply(mat, 1, function(row) sort(row, decreasing = FALSE))
#   ref <- colMeans(smat)
# 
#   yy <- mat
#   yy[yy == 0] = NA
# 
#   refS <- sort(ref)
# 
#   k <- which(refS > 0)[1]
#   lo <- (length(refS) - k + 1)
# 
#   if (qFlag) {
#     diffr <- apply(yy, 1, function(row) {
#       refS[k:length(refS)] - quantile(row, probs = seq(0, 1, length.out = lo), na.rm = TRUE)
#     })
#   } else {
#     diffr <- apply(yy, 1, function(row) {
#       refS[k:length(refS)] - approx(sort(row, decreasing = FALSE), n = lo)$y
#     })
#   }
# 
#   diffr2 <- matrixStats::colMedians(abs(diffr), na.rm = TRUE)
# 
#   x <- which(abs(diff(diffr2)) / diffr2[-1] > rel)[1] / length(diffr2)
#   if (x <= 0.50) {
#     message("Default value being used.")
#     x <- 0.50
#   }
# 
#   return(x)
# }
# 
# cumNormStatFast <- function(mat, pFlag = FALSE, rel = .1, ...) {
# 
#   smat <- apply(mat, 1, function(row) {
#     sort(row[which(row > 0)], decreasing = TRUE)
#   })
# 
#   leng <- max(sapply(smat, length))
#   if (any(sapply(smat, length) == 1)) stop("Warning: sample with one or zero features")
# 
#   smat2 <- array(NA, dim = c(leng, nrow(mat)))
#   for (i in 1:nrow(mat)) {
#     smat2[leng:(leng - length(smat[i]) + 1), i] = smat[i]
#   }
# 
#   rmat2 <- apply(smat2, 1, function(row) {
#     quantile(row, probs = seq(0, 1, length.out = ncol(smat2)), na.rm = TRUE)
#   })
# 
#   smat2[is.na(smat2)] = 0
#   ref1 <- colMeans(smat2)
# 
#   ncols <- ncol(rmat2)
#   diffr <- apply(rmat2, 1, function(row) {
#     ref1 - row
#   })
#   diffr1 <- matrixStats::colMedians(abs(diffr))
# 
#   x <- which(abs(diff(diffr1)) / diffr1[-1] > rel)[1] / length(diffr1)
#   if (x <= 0.50) {
#     message("Default value being used.")
#     x <- 0.50
#   }
# 
#   return(x)
# }


###############
## setup the MRexperiment object
## phenodat: sample data.frame with rownames being the
## sampleID (crucial), first column also the sampleID here
## OTUdata: feaure data.frame with rownames being the featID
## (crucial). first column also the featID
## cntdat: matrix data.frame with rownames being the featID and
## colnames as sampleID
###############

setMRobject <- function(cntdat, phenodat, featdat) {
  suppressPackageStartupMessages(require("metagenomeSeq"))
  phenodatDF <- as.data.frame(phenodat)
  phenotypeData <- AnnotatedDataFrame(phenodatDF)

  featdatDF <- as.data.frame(featdat)
  OTUdata <- AnnotatedDataFrame(featdatDF)

  cntdatDF <- as.data.frame(cntdat)

  obj <- newMRexperiment(cntdatDF, phenoData = phenotypeData, featureData = OTUdata)
  return(obj)
}

###############
  ### Filter OTU/Taxa presence with a chosen read count
  ### 'present' for the number of samples the feature is at least present in
  ### 'fdepth' for feature count cutoff for presence, a taxa/OTU has to have
  ### at least this number of reads to be deemed present within a sample.
  ### 'depth' for the total number reads a sample has to at least have
###############
filterMRobject <- function(obj, present = 1, fdepth = 1, depth = 1, norm = FALSE) {
  mat <- returnAppropriateObj(obj, norm = norm, log = FALSE) > (fdepth - 1)
  cols <- which(colSums(MRcounts(obj, norm = norm)) >= depth)
  rows <- which(rowSums(mat[, cols]) >= present)

  # Apply filter
  obj <- obj[rows, cols]
  return(obj)
}


###############
# Normalize the data with percentile param
# When percentile is NULL, normalization factor will be calculated
###############
normMRobject <- function(obj, percentile = NULL) {
  # Calculating normalization factor
  if (is.null(percentile)) {
    percentile <- cumNormStat(obj)

  }
  # Apply normalization
  obj <- cumNorm(obj, p = signif(percentile, 4))
  return(obj)
}

###############
# plot histograms to get a sense of the counts and presence
###############
plotHistMeta <- function(x, xlab="log2(Sum of counts)", main="Histogram", breaks=50) {
  hist(x, xlab = xlab, main = main, breaks = breaks)
}

##################
# Boxplot of distributions: Before and after normalization in log2 form
##################
boxplotMeta <- function(obj, keyAnnot, cols = NULL) {
  #color setup
  if (is.null(cols)) {
    suppressPackageStartupMessages(require("RColorBrewer"))
    cols <- c(brewer.pal(8, "Accent"), rev(brewer.pal(8, "Dark2")[-8]), brewer.pal(8, "Paired"))
  }

  cl <- factor(pData(obj)[, keyAnnot])
  clcol <- cols[as.integer(cl)]

  #plot
  par(mfrow = c(2, 1))
  boxplot(log2(1 + MRcounts(obj, norm = F, log = F)), col = clcol, outcol = clcol, ylab = "log2(1+Abundance)")
  boxplot(log2(1 + MRcounts(obj, norm = T, log = F)), col = clcol, outcol = clcol, ylab = "log2(1+Abundance)")
}



##################
# PCoA - Bray-Curtis
##################
fitPCoA <- function(obj, method = "bray", norm = T) {
  suppressPackageStartupMessages(library("vegan"))
  suppressPackageStartupMessages(library("ape"))

  #### distance computation and dimension reduction
  d <- vegdist(t(MRcounts(obj, norm = norm, log = F)), method = method)
  pcodecomp <- pcoa(d)

  # if any eigenvalue is negative, leverage Cailliez correction
  if (sum(pcodecomp$values$Relative_eig < 0) > 0) {
    pcodecomp <- pcoa(d, correction = "cailliez")
  }
  return(pcodecomp)
}

##################
# Plot PCoA - Bray-Curtis
##################
plotPCoA <- function(obj, pcodecomp, keyAnnot, keyAnnot2=NULL, dimn=2, fileNameAdd="", cols=NULL){
  suppressPackageStartupMessages(require(ape))
  
  #color setup
  if(is.null(cols)){
    suppressPackageStartupMessages(require("RColorBrewer"))
    cols = c(brewer.pal(8, "Accent"),rev(brewer.pal(8, "Dark2")[-8]), brewer.pal(8,"Paired"))
  }
  cl=cl2=factor(pData(obj)[,keyAnnot])
  clcol=cols[as.integer(cl)]
  
  # if a secondary annotation were specified, pch is used
  if(!is.null(keyAnnot2)){
    cl2 <- factor(pData(obj)[,keyAnnot2])
  }
  pch2use=c(1: length(levels(cl2)))
  pchInput=pch2use[cl2]
  
  # Compute the percentage of explained variance
  PCOAaxes <- pcodecomp$vectors[,c(1:dimn)]
  eignPERC<- pcodecomp$values$Rel_corr_eig[c(1:dimn)]
  
  # plot PCoA
  pairs(PCOAaxes, main=paste("PCoA", fileNameAdd), col=clcol, pch=pchInput, 
        cex=1.1, cex.labels = 2, cex.axis=1.5, upper.panel=NULL,
        labels=paste("Dim",1:dimn,"\n",round(eignPERC,3)*100,"%"))  
  if(is.null(keyAnnot2)){
    legend("topright", legend = levels(cl), col=cols, pch=pch2use, ncol=3, cex=1)#, inset = c(0.1, 0.1)) 
  }else{
    legend("top", legend = levels(cl), col=cols, pch = 1, ncol=3, cex=1, inset = c(0.1, 0.1)) 
    legend("topright", legend = levels(cl2), pch=pch2use, cex=1, inset = c(0.1, -0.1)) 
  }
  
}
