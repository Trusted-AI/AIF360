/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.ibm.aif360.processors.generic;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.io.IOUtils;
import org.apache.nifi.annotation.behavior.ReadsAttribute;
import org.apache.nifi.annotation.behavior.ReadsAttributes;
import org.apache.nifi.annotation.behavior.WritesAttribute;
import org.apache.nifi.annotation.behavior.WritesAttributes;
import org.apache.nifi.annotation.documentation.CapabilityDescription;
import org.apache.nifi.annotation.documentation.SeeAlso;
import org.apache.nifi.annotation.documentation.Tags;
import org.apache.nifi.annotation.lifecycle.OnScheduled;
import org.apache.nifi.components.PropertyDescriptor;
import org.apache.nifi.flowfile.FlowFile;
import org.apache.nifi.processor.AbstractProcessor;
import org.apache.nifi.processor.ProcessContext;
import org.apache.nifi.processor.ProcessSession;
import org.apache.nifi.processor.ProcessorInitializationContext;
import org.apache.nifi.processor.Relationship;
import org.apache.nifi.processor.exception.ProcessException;
import org.apache.nifi.processor.io.InputStreamCallback;
import org.apache.nifi.processor.util.StandardValidators;

@Tags({ "example" })
@CapabilityDescription("Provide a description")
@SeeAlso({})
@ReadsAttributes({ @ReadsAttribute(attribute = "", description = "") })
@WritesAttributes({ @WritesAttribute(attribute = "", description = "") })
public class MyProcessor extends AbstractProcessor {

    public static final PropertyDescriptor COLUMNS = new PropertyDescriptor.Builder().name("COLUMNS_PROPERTY")
            .displayName("Columns").description("Comma separated list of column names").required(true)
            .addValidator(StandardValidators.NON_EMPTY_VALIDATOR).build();

    public static final PropertyDescriptor PROTECTED_ATTRIBUTE_NAMES = new PropertyDescriptor.Builder()
            .name("PROTECTED_ATTRIBUTE_NAMES").displayName("Protected")
            .description("Comma separated list of protected attribute names").required(true)
            .addValidator(StandardValidators.NON_EMPTY_VALIDATOR).build();

    public static final PropertyDescriptor GROUND_TRUTH_TARGET_NAMES = new PropertyDescriptor.Builder()
            .name("GROUND_TRUTH_TARGET_NAMES").displayName("Ground truth target names")
            .description("Comma separated list of ground truth target column names").required(true)
            .addValidator(StandardValidators.NON_EMPTY_VALIDATOR).build();

    public static final PropertyDescriptor PREDICTED_TARGET_NAMES = new PropertyDescriptor.Builder()
            .name("PREDICTED_TARGET_NAMES").displayName("Predicted target names")
            .description("Comma separated list of ground predicted target column names").required(true)
            .addValidator(StandardValidators.NON_EMPTY_VALIDATOR).build();

    public static final PropertyDescriptor PRIVILEGED_GROUPS = new PropertyDescriptor.Builder()
            .name("PRIVILEGED_GROUPS").displayName("Privileged group column names and values")
            .description(
                    "Comma separated list of key:value pairs separated by colon, key=>column name, value=>privileged group value/id")
            .required(true).addValidator(StandardValidators.NON_EMPTY_VALIDATOR).build();
    public static final PropertyDescriptor UNPRIVILEGED_GROUPS = new PropertyDescriptor.Builder()
            .name("UNPRIVILEGED_GROUPS").displayName("Unprivileged group column names and values")
            .description(
                    "Comma separated list of key:value pairs separated by colon, key=>column name, value=>unprivileged group value/id")
            .required(true).addValidator(StandardValidators.NON_EMPTY_VALIDATOR).build();

    public static final Relationship SUCCESS_RELATIONSHIP = new Relationship.Builder().name("success")
            .description("success relationship").build();

    public static final Relationship FAILURE_RELATIONSHIP = new Relationship.Builder().name("failure")
            .description("failure relationship").build();

    private List<PropertyDescriptor> descriptors;

    private Set<Relationship> relationships;

    @Override
    protected void init(final ProcessorInitializationContext context) {
        final List<PropertyDescriptor> descriptors = new ArrayList<PropertyDescriptor>();
        descriptors.add(COLUMNS);
        descriptors.add(PROTECTED_ATTRIBUTE_NAMES);
        descriptors.add(GROUND_TRUTH_TARGET_NAMES);
        descriptors.add(PREDICTED_TARGET_NAMES);
        descriptors.add(PRIVILEGED_GROUPS);
        descriptors.add(UNPRIVILEGED_GROUPS);
        this.descriptors = Collections.unmodifiableList(descriptors);

        final Set<Relationship> relationships = new HashSet<Relationship>();
        relationships.add(SUCCESS_RELATIONSHIP);
        relationships.add(FAILURE_RELATIONSHIP);
        this.relationships = Collections.unmodifiableSet(relationships);
    }

    @Override
    public Set<Relationship> getRelationships() {
        return this.relationships;
    }

    @Override
    public final List<PropertyDescriptor> getSupportedPropertyDescriptors() {
        return descriptors;
    }

    @OnScheduled
    public void onScheduled(final ProcessContext context) {

    }

    @Override
    public void onTrigger(final ProcessContext context, final ProcessSession session) throws ProcessException {

        String columns = context.getProperty(COLUMNS.getName()).getValue();
        String protected_attribute_names = context.getProperty(PROTECTED_ATTRIBUTE_NAMES).getValue();
        String ground_truth_target_names = context.getProperty(GROUND_TRUTH_TARGET_NAMES).getValue();
        String predicted_target_names = context.getProperty(PREDICTED_TARGET_NAMES).getValue();
        String privileged_groups = context.getProperty(PRIVILEGED_GROUPS).getValue();
        String unprivileged_groups = context.getProperty(UNPRIVILEGED_GROUPS).getValue();

        FlowFile flowFile = session.get();
        if (flowFile == null) {
            return;
        }

        try {
            String[] returnValue = { "error_during_processing" }; // using String[] over String to mitigate limitation
                                                                  // of Java that variables in an enclosing scope can't
                                                                  // be changed in an inner class
            session.read(flowFile, new InputStreamCallback() {
                @Override
                public void process(InputStream in) throws IOException {
                    try {
                        String rows = IOUtils.toString(in);
                        String tmpFile = "/tmp/deleteme" + System.currentTimeMillis();
                        BufferedWriter writer = new BufferedWriter(new FileWriter(tmpFile, true));
                        writer.append(rows);
                        writer.close();

                        Process p = Runtime.getRuntime().exec(
                                "/Users/romeokienzler/Documents/proj/1codait/lfai/lfai_nifi/360/run_java_script.sh "
                                        + tmpFile + " " + columns + " " + protected_attribute_names + " "
                                        + ground_truth_target_names + " " + predicted_target_names + " "
                                        + privileged_groups + " " + unprivileged_groups);

                        BufferedReader stdInput = new BufferedReader(new InputStreamReader(p.getInputStream()));

                        BufferedReader stdError = new BufferedReader(new InputStreamReader(p.getErrorStream()));

                        // read the output from the command
                        String s = null;
                        System.out.println("Here is the standard output of the command:\n");
                        while ((s = stdInput.readLine()) != null) {
                            System.out.println(s);
                            returnValue[0] = s;
                        }

                        

                        // read any errors from the attempted command
                        System.out.println("Here is the standard error of the command (if any):\n");
                        while ((s = stdError.readLine()) != null) {
                            System.out.println(s);
                        }


                    } catch (Exception ex) {
                        ex.printStackTrace();
                        getLogger().error("Failed to read json string.");
                    }
                }
            });

            System.out.println("result");
            System.out.println(returnValue[0]);
            flowFile = session.putAttribute(flowFile, "aif360.result", returnValue[0]);
            session.transfer(flowFile, SUCCESS_RELATIONSHIP);
        } catch (ProcessException e) {
            session.transfer(flowFile, FAILURE_RELATIONSHIP);
        }

    }
}
