FROM openfoam/openfoam7-paraview56

# Install python3.6
#RUN yum makecache fast
#RUN yes | yum install centos-release-scl
#RUN yes | yum install -y rh-python36

# Copy openfoam bashrc file
#RUN echo "source /opt/OpenFOAM/OpenFOAM-v1812/etc/bashrc" >> /etc/bashrc
#RUN cp -r /opt/OpenFOAM/OpenFOAM-v1812/etc/* /etc/

# Create a new user called foam
#RUN sudo useradd --user-group --create-home --shell /bin/bash foam
#RUN echo "foam ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

#RUN echo "source /opt/OpenFOAM/OpenFOAM-v1812/etc/bashrc" >> ~foam/.bashrc
ENV LD_LIBRARY_PATH $WM_THIRD_PARTY_DIR/platforms/linux64Gcc/ParaView-5.5.2/lib/mesa:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH $WM_THIRD_PARTY_DIR/platforms/linux64Gcc/ParaView-5.5.2/lib/paraview-5.5/plugins:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH $WM_THIRD_PARTY_DIR/platforms/linux64Gcc/qt-5.9.0/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH $WM_THIRD_PARTY_DIR/platforms/linux64/zlib-1.2.11/lib:$LD_LIBRARY_PATH
ENV PATH $WM_THIRD_PARTY_DIR/platforms/linux64Gcc/qt-5.9.0/bin:$PATH
ENV QT_PLUGIN_PATH $WM_THIRD_PARTY_DIR/platforms/linux64Gcc/qt-5.9.0/plugins
#ENV DISPLAY :0

#USER foam

USER root
RUN apt-get update
RUN yes | apt-get install python3-pip
RUN yes | apt-get install gmsh
USER foam
COPY requirements.txt /home/openfoam
RUN pip3 install -r requirements.txt