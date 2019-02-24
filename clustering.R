[10:23 AM, 2/24/2019] Mayur Bansal: normalized_median1 = (d2_cluster$sale_amt_median-min(d2_cluster$sale_amt_median))/(max(d2_cluster$sale_amt_median)-min(d2_cluster$sale_amt_median))

d2_clust_upd<-mutate(d2_cluster,norm=normalized_median1)

d2_clust_upd<-d2_clust_upd[,c(1,4)]

library(mclust)
clusters_mclust_upd = Mclust(d2_clust_upd)
summary(clusters_mclust_upd)

mclust_bic_upd = -sapply(1:10,FUN = function(x) Mclust(d2_clust_upd,G=x)$bic)
mclust_bic_upd
library(ggplot2)
ggplot(data=data.frame(cluster = 1:10,bic = mclust_bic_upd),aes(x=cluster,y=bic))+
  geom_line(col='steelblue',size=1.2)+
  geom_point()+
  scale_x_continuous(breaks=seq(1,10,1))

m_clusters_upd = Mclust(data = d2_clust_upd,G = 4)
m_clusters_upd

m_segments_upd = m_clusters_upd$classification
table(m_segments_upd)

library(cluster)
clusplot(d2_clust_upd,
         m_segments_upd)

cluster_dwelling<-cbind(d2_clust_upd,m_segments_upd)
[10:24 AM, 2/24/2019] Mayur Bansal: normalized_median2 = (d3_cluster$sale_amt_median-min(d3_cluster$sale_amt_median))/(max(d3_cluster$sale_amt_median)-min(d3_cluster$sale_amt_median))

d3_clust_upd<-mutate(d3_cluster,norm=normalized_median2)
d3_clust_upd<-d3_clust_upd[,c(1,3)]

clusters_mclust_city = Mclust(d3_clust_upd)
summary(clusters_mclust_city)

mclust_bic_city = -sapply(1:10,FUN = function(x) Mclust(d3_clust_upd,G=x)$bic)
mclust_bic_city
library(ggplot2)
ggplot(data=data.frame(cluster = 1:10,bic = mclust_bic_city),aes(x=cluster,y=bic))+
  geom_line(col='steelblue',size=1.2)+
  geom_point()+
  scale_x_continuous(breaks=seq(1,10,1))

m_clusters_city = Mclust(data = d3_clust_upd,G = 9)
m_clusters_city

m_segments_city= m_clusters_city$classification
table(m_segments_city)

clusplot(d3_clust_upd,
         m_segments_city)

cluster_city<-cbind(d3_clust_upd,m_segments_city)
filter(cluster_city,m_segments_city==9)
filter(cluster_dwelling,m_segments_upd==3)
filter(cluster_dwelling,m_segments_upd==1)
cluster_city[,c(1,3)]
write.csv(cluster_city,'cluster_city.csv')
