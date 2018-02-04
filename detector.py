# making a Hog and Linear SVM classifier on our own is a tedious task
  # So we will insted use dlib package which has an API for creating such custom object detectors
  # So insted we will create an abstraction to use object detector from dlib with ease
  # the actual functioning of Hog and linear SVM will be broken down into simple steps

  # TRAINING :
  # create a hog descripter with certain pixels_per_cells, cells_per_block and orientations
  # Extract hog features from the object region
  # create and train Linear SVM model on the extracted features of hog

  # TESTING :
  # estimate the average window size
  # Scale up or down the images for several levels upto a certain termination and build an image pyramid
  # slide the window in each image in a image pyramid
  # extract hog features from each location
  # estimate the probability of trained SVM model with the current hog features
  # it is important to note that we are going to calculate the PROBABILITY OF TRAINED SVM MODEL WITH CURRENT HOG FEATURES  AND IF IT IS MORE THAN A CERTAIN THERSHOLD THEN IT CONTAINS THE OBJECT OTHERWISE NOT

# detector.py

  import dlib
  import cv2

  class ObjectDetector(object):
      def __init__(self, options = None, loadPath = None):
          # create detector options
          self.options = options
          # default options will be used if no options are provided explicitely
          # these options consists of several hyperparameters like window_size, num_threads,
          if self.options is None:
              self.options = dlib.simple_object_detector_training_options()

          if loadPath is not None:
              self._detector = dlib.simple_object_detector(loadPath)

## in above the constructor takes 2 arguments :
# options: object detector options for controlling HOG and SVM hyperparameters
# loadPath: to load the trained detector from the path
       def _prepare_annotations(self, annotations):
           annots = []
           for (x, y, xb, yb) in annotations:
               annots.append([dlib.rectangle(left = long(x), top = long(y), right = long(xb), bottom = long(yb))])
               return annots

       def _prepare_images(self, imagePaths):
           images = []
           for imPath in imagePaths:
               image = cv2.imread(imPath)
               image = cv2.cvtColor("image", COLOR_BGR2RGB)
               image.append(image)
           return images
# these 32 packages help in preprocessing the given annotations to the form that is acceptable by dlib detector
# dlib expects the image in BGR format
       def fit(self, imagePaths, annotations, visulaize = False, savePath = None):
            annotations = self._prepare_annotations(annotations)
            images = self._prepare_images(imagePaths)
            self._detector = dlib.train_simple_object_detector(image, annotations, self.options)

            # visualize Hog
            if visualize:
                win = dlib.image_window()
                win.set_image(self._detector)
                dlib.hit_enter_to_continue()

                # save the detector to the disk
            if savePath is not None:
                self._detector.save(Path)

            return self
# our fit method takes the arguments in as follows:
# annotations- a numpy array containing annotations corresponding to images in the imagePaths
# imagePaths- a numpy array of type unicode containing paths to the images
# visulaize- a flag indicating whether or not to visulaize the trained hog features
# savePath- path to save the trained detector, if none then no detector will be saved

        def predict(self, image):
            boxes = self._detector(image)
            preds = []
            for box in boxes:
                (x, y, xb, yb) = [box.left(), box.top(), box.right(), box.bottom()]
                preds.append((x, y, xb, yb))
            return preds

        def detect(self, image, annotate = None):
            image = cvtColor("image", COLOR_RGB2BGR)
            pred = self.predict(image)
            for (x, y, w, h) in image:
                image = cvtColor("image", COLOR_RGB2BGR)

                #draw and annotate the image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                if annotate is not None and type(annotate) == str:
                    cv2.putText(image, annotate, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 255, 0), 2)
            cv2.imshow("Detected", image)
            cv2.waitKey(0)
