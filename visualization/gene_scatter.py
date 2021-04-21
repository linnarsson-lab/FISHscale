import matplotlib.pyplot as plt



colors_new = pkl.load(open('oPool7_RK_gene_colors/oPool7_RK_gene_colors_V3.pkl', 'rb'))
colors_new = [colors_new[g] for g in RK_gene_order]
    
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)#, rasterized=True)
ax.set_rasterization_zorder(1)

#Sort on Ripley K so that the most self clustered genes are plotted last, and thus on top. 
for i, g in enumerate(RK_gene_order):
    ax.scatter(gene_coord[g]['c_stitched_coords'], gene_coord[g]['r_stitched_coords'], s=0.1, color=colors_new[i], zorder=0)
    
ax.set_aspect('equal')
ax.set_axis_off()
ax.add_patch(plt.Rectangle((0,0), 1, 1, facecolor=(0,0,0),
                           transform=ax.transAxes, zorder=-1))

#Add scale bar
pxsize = 0.18 #um
scalebar_size = 1000
npix = scalebar_size / pxsize
plt.hlines(0, 0, npix, colors=['red'])
plt.text(0,0, f'{scalebar_size}um', c='white')

plt.tight_layout()

#Plot limits of the 100% image
x_100p = plt.xlim()
y_100p = plt.ylim()
